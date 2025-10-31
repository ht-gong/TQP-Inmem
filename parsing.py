import io
import os
import sys
import shutil
import json

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, DateType, StructField, IntegerType, StringType, DoubleType

from expression import parse
from dataclasses import dataclass, field
import re
from enum import Enum

def get_id(name): # name#id --> id
  return int(name.split('#')[-1].replace('L', ''))

def get_name(name): # name#id --> name
  return name.split('#')[0]

def get_literals(tree, literal_set):
  # print('tree', tree)
  if isinstance(tree, str):
    if '#' in tree and not tree.startswith('Brand#'): 
      if 'as' in tree:
        tree = tree.split(' as', 1)[0]
        literal_set.add(get_id(tree))
      else:
        literal_set.add(get_id(tree))

  if isinstance(tree, list) and tree[0] == 'isnotnull':
    return 
  
  if isinstance(tree, list) or isinstance(tree, tuple):
    if len(tree) == 3 and tree[1] == '#':
      literal_set.add(int(tree[-1]))
      get_literals(tree[0], literal_set)
    else:
      for e in tree:
        get_literals(e, literal_set)

def right_most(tree):
  if isinstance(tree, list):
    return right_most(tree[-1])
  return tree

def extract_number_after_hash(s):
  match = re.search(r'#(\d+)', s)
  return int(match.group(1)) if match else None

def get_project_output_ids(expr_tree):
    output_ids = []
    for tree in expr_tree:
      while isinstance(tree, list) and len(tree) == 1:
        tree = tree[0]
      output_ids.append(get_id(right_most(tree)))
    return output_ids

def get_project_used_cols(expr_tree, available_cols_set):
    rename_map = {}
    literal_set = set()
    material_out, used_expr = [], []
    for tree in expr_tree:
        while isinstance(tree, list) and len(tree) == 1:
            tree = tree[0]
        renamed_id = get_id(right_most(tree))
        if not isinstance(tree, str):
            if tree[1] == 'AS' and isinstance(tree[0], str): ### rename
                rename_map[get_id(tree[0])] = renamed_id
            else:
                get_literals(tree, literal_set)
                material_out.append(renamed_id)
                used_expr.append(tree)
    
    return rename_map, literal_set & available_cols_set, material_out, used_expr

def get_aggregation_ops(tree):
  if isinstance(tree, str):
    if tree == 'sum' or tree == 'min' or tree == 'max' or tree == 'count' or tree == 'avg':
      return tree
    return ''
  if isinstance(tree, list):
    for e in tree:
      r = get_aggregation_ops(e)
      if r != '':
        return r
    return ''

def get_aggregation_key_and_values(expr_tree):
    input_key_ids, output_key_ids, output_agg_ids, agg_ops = [], [], [], []
    for tree in expr_tree:
        while isinstance(tree, list) and len(tree) == 1:
          tree = tree[0]
        if isinstance(tree, str) or (len(tree) == 3 and tree[1] == 'AS' and isinstance(tree[0], str)): # Key
          input_key_ids.append(get_id(tree if isinstance(tree, str) else tree[0]))
          output_key_ids.append(get_id(tree if isinstance(tree, str) else tree[-1])) # MAY REASSIGN ID
        else:
          output_agg_ids.append(get_id(right_most(tree)))
          agg_ops.append(get_aggregation_ops(tree))
    return input_key_ids, output_key_ids, output_agg_ids, agg_ops

def is_simple_projection(expr_trees):
    used_set = set()
    for tree in expr_trees:
        while isinstance(tree, list) and len(tree) == 1:
            tree = tree[0]

        agg_id = get_id(right_most(tree))
        if isinstance(tree, str):
            used_set.add(agg_id)
        else:
            if tree[1] == 'AS' and isinstance(tree[0], str):
                used_set.add(get_id(tree[0]))
            else:
                return (False, set())
    return (True, used_set)


@dataclass
class QueryPlanRep:
    ops_idx: list = field(default_factory=list)
    ops_dict: dict = field(default_factory=dict)
    query_parent_map: dict = field(default_factory=dict)  # child to parent mapping
    query_child_map: dict = field(default_factory=dict)  # parent to child mapping
    in_col_dict: dict = field(default_factory=dict)
    out_col_dict: dict = field(default_factory=dict)

class MaterialStrat(Enum):
    DISABLED = 1
    SIMPLE = 2
    BEST_EFFORT = 3

class ParsingLayer:
    MASK_COL_START = 10_000

    def __init__ (self):
        with open('./queries/schema.json', 'r') as f:
            schema_json = json.load(f)

        self.spark = SparkSession.builder   \
        .config("spark.local.dir", "./tmp") \
        .getOrCreate()
        
        # .config("spark.driver.memory", "1g") \
        # .config("spark.executor.memory", "1g") \
        # .config("spark.executor.memoryOverhead", "1g") \
        
        self.schema = {}
        type_mapping = {
            "identifier": IntegerType(),
            "integer": IntegerType(),
            "decimal": DoubleType(),
            "date": DateType(),
            "variable text": StringType(),
            "fixed text": StringType()
        }
        for table_name, table_schema in schema_json.items():
            fields = []
            for column_name, column_type in table_schema.items():
                if column_name != 'PRIMARY_KEY':
                    fields.append(StructField(column_name.lower(), type_mapping[column_type.split(',')[0]], False))

            self.schema[table_name.lower()] = StructType(fields)
            
        self.df = {}

        self.mask_col_id = self.MASK_COL_START

        # Make sure the dataset is parquet (column store), not RDD (row store)
        # Scan parquest will discard unrelated columns

        for table, table_schema in self.schema.items():
            df_empty = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD(), table_schema)
            output_path = f"./tmp/tmpfile-{table}"
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            df_empty.write.parquet(output_path)
            self.df[table] = self.spark.read.parquet(output_path)
            self.df[table].createOrReplaceTempView(table)
    
    def get_next_mask_id(self, used_set):
        while self.mask_col_id in used_set:
            self.mask_col_id += 1
        ret = self.mask_col_id
        self.mask_col_id += 1
        return ret

    def generate(self, query, qid):
        """
        Generates a parsed representation of a Spark SQL query's physical plan.
        This method executes the given SQL query using Spark, captures the physical
        plan in a formatted string, and then parses this plan into a tree structure
        and a set of arguments.
        Args:
            query (str): The SQL query to be executed and analyzed.
        Returns:
            tuple: A tuple containing:
                - int: The number of operations in the physical plan.
                - dict: A dictionary of arguments for each node, where keys are node
                  IDs and values are dictionaries of argument names and their values.
        """
        result = self.spark.sql(query)
        # result.explain('formatted')

        def capture_explain(result):
            captured_output = io.StringIO()
            sys.stdout = captured_output
            result.explain(mode='formatted')
            sys.stdout = sys.__stdout__
            return captured_output.getvalue()
        
        # Capture physical plan of the query as a string
        plan_string = capture_explain(result)

        print(plan_string)

        def split_plan(plan_lines):
            tree_start_pos = next(i for i, line in enumerate(plan_lines) if line.startswith("+-"))
            tree_lines = plan_lines[tree_start_pos:]
            tree_end_pos = next(i for i, line in enumerate(tree_lines) if not line.strip())
            item_lines = tree_lines[tree_end_pos:]
            tree_lines = tree_lines[:tree_end_pos]
            return tree_lines, item_lines
        
        # Split the physical plan into "tree" and "item" parts
        subquery_mark = '===== Subqueries ====='
        sub_tree_lines, sub_item_lines = None, None
        if subquery_mark in plan_string:
            plan_string, sub_plan_string = plan_string.split(subquery_mark) # Throw it away
            sub_plan_lines = sub_plan_string.split("\n")[3:]
            sub_tree_lines, sub_item_lines = split_plan(sub_plan_lines)
            
        plan_lines = plan_string.strip().split("\n")
        tree_lines, item_lines = split_plan(plan_lines)


        class TreeNode:
            def __init__(self, id, depth):
                self.id = id
                self.depth = depth
                self.children = []

            def add_child(self, child_node):
                self.children.append(child_node)

        def parse_tree(lines, children, args):
            root = None
            stack = []
            for line in lines:
                depth = len(line) - len(line.lstrip(" -:+"))
                id = int(line[depth:-1].split('(')[-1])
                node = TreeNode(id, depth)
                if 'BuildRight' in line:
                    args[id]['build'] = 'right'
                elif 'BuildLeft' in line:
                    args[id]['build'] = 'left'
                
                if root == None:
                    root = node
                    stack = [node]
                else:
                    while stack and stack[-1].depth >= depth:
                        stack.pop()
                    stack[-1].add_child(node)
                    children.setdefault(stack[-1].id, []).append(node.id)
                    stack.append(node)

        def parse_item(item_lines):
            ops = []
            args = {}
            op = None
            for line in item_lines:
                if not line:
                    continue
                if line.startswith('('): # (ID) Operator Name
                    op = int(line[1:].split(')')[0])
                    ops.append(op)
                    name = line.split(')')[1].strip()
                    args[op] = {'name': name}
                else: # [key] : value
                    # Results [1]: [((100.0 * sum(CASE WHEN StartsWith(p_type#31, PROMO) THEN (l_extendedprice#243 * (1.0 - l_discount#244)) ELSE 0.0 END)#306) / sum((l_extendedprice#243 * (1.0 - l_discount#244)))#307) AS promo_revenue#305]
                    if line.startswith('Subquery:2'):
                        break
                    key = line.split('[')[0].split(':')[0].strip()

                    sp = 'Results [1]:'
                    # This is a special case for TPC-H Q15, where a comma is present inside a substring function argument list.
                    # The default comma-based splitting logic fails for this line.
                    # Example line: ...: [substring(c_phone#133, 1, 2) AS cntrycode#305, c_acctbal#134]
                    match = re.search(r'substring\(c_phone#(\d+), 1, 2\) AS cntrycode#(\d+), (c_acctbal#\d+)\]$', line)
                    if match:
                        args[op][key] = [f'substring(c_phone#{match.group(1)}, 1, 2) AS cntrycode#{match.group(2)}', match.group(3)]
                    elif line.startswith(sp):
                        args[op][key] = [line[len(sp):].strip('[ ]')]
                    elif key == 'Condition' or key == 'Join condition':
                        args[op][key] = [line.split(':')[1].strip()]
                    else:
                        args[op][key] = [item.strip() for item in line.split(':', 1)[1].strip('[ ]').split(',')]
            
            for u in list(args.keys()):
                args_u = args[u]
                name = args_u['name']

                if name == 'Scan parquet':
                    args_u['name'] = 'Scan'
                    # Location: InMemoryFileIndex [file:../tmp/tmpfile-lineitem]
                    args_u['table name'] = args_u['Location'][-1].split('-')[-1].upper()
                    # Output [7]: [l_quantity#242, l_extendedprice#243, l_discount#244, l_tax#245, l_returnflag#246, l_linestatus#247, l_shipdate#248]
                    args_u['output names'] = [item.split('#')[0] for item in args_u['Output']]
                    args_u['output ids'] = [int(item.split('#')[1]) for item in args_u['Output']]
                    args_u.pop('Output')

                elif name == 'Filter':
                    args_u['input ids'] = [int(col.split('#')[-1]) for col in args_u.pop('Input')]
                    args_u['filter tree'] = parse(args_u.pop('Condition')[0])
                    args_u['output ids'] = args_u['input ids'].copy()

                elif name == 'Project':
                    if args_u['Output'] == ['']:
                        args_u['output'] = []
                    else:
                        args_u['output'] = [parse(col) for col in args_u.pop('Output')]
                    args_u['input ids'] =  [get_id(col) for col in args_u.pop('Input')]
                    args_u['output ids'] = get_project_output_ids(args_u['output'])
                
                elif name.endswith('Aggregate'):
                    args_u['name'] = 'Aggregate'
                    args_u['is partial'] = args_u.pop('Functions')[0].startswith("partial")
                    if args_u['Keys'] != ['']:
                        args_u['keys'] = args_u.pop('Keys')
                    else:
                        args_u['keys'] = None

                    args_u['input ids'] =  [get_id(col) for col in args_u.pop('Input')]
                    args_u['results'] = [parse(col) for col in args_u.pop('Results')]
                    args_u['in key ids'], args_u['out key ids'], args_u['out agg ids'], args_u['agg ops'] = get_aggregation_key_and_values(args_u['results'])
                    args_u['output ids'] = args_u['out key ids'] + args_u['out agg ids']
                 
                elif name.endswith('Exchange'):
                    args_u['name'] = 'Exchange'
                
                elif name == 'Sort':
                    args_u['input ids'] = [get_id(col) for col in args_u.pop('Input')]
                                # args_u['Arguments'],
                    # # [l_returnflag#56 ASC NULLS FIRST, l_linestatus#57 ASC NULLS FIRST], true, 0
                    keys = args_u.pop('Arguments')[:-2]
                    args_u['key ids']  = [get_id(key.split(' ')[0]) for key in keys]
                    args_u['key orders'] = [key.split(' ')[1] == 'DESC' for key in keys] # ASC or DESC
                    args_u['output ids'] = args_u['input ids'].copy()

                elif name == 'TakeOrderedAndProject':
                    args_u['name'] = 'take ordered'
                    args_u['input ids'] = [get_id(col) for col in args_u.pop('Input')]
                    # Arguments: 10, "[revenue#305 DESC NULLS LAST, o_orderdate#111 ASC NULLS FIRST]", "[l_orderkey#48, revenue#305, o_orderdate#111, o_shippriority#114]"
                    ag_str = args_u.pop('Arguments')
                    indices = [i for i, s in enumerate(ag_str) if '[' in s]

                    # Reconstruct the key and output strings based on bracket positions
                    args_u['limit'] = int(ag_str[0])
                    keys_str = ag_str[indices[0]: indices[1]]
                    outputs_str = ag_str[indices[1]:]
                    args_u['key ids'] = [get_id(item.strip().split(' ')[0]) for item in keys_str]
                    args_u['key orders']  = [item.strip().split('#')[-1].split(' ')[1] == 'DESC' for item in keys_str]
                    args_u['output ids'] = [get_id(item.strip()) for item in outputs_str]

                elif name.endswith('Join'):
                    args_u['name'] = 'Join'
                    args_u['left keys'] = [extract_number_after_hash(ele) for ele in args_u.pop('Left keys')]
                    args_u['right keys'] = [extract_number_after_hash(ele) for ele in args_u.pop('Right keys')]
                    args_u['join type'] = args_u.pop('Join type')[0]

                    if 'Join condition' in args_u.keys():
                        if args_u['Join condition'][0] == 'None':
                            args_u.pop('Join condition')
                        else:
                            args_u['join condition'] = parse(args_u.pop('Join condition')[0])
                
                elif name == 'AdaptiveSparkPlan':     # Treated as output operator
                    args_u['output ids'] = [get_id(col) for col in args_u['Output']]
                    args_u['output names'] = [get_name(col) for col in args_u['Output']]
                    args_u.pop('Output')
                
            return ops, args
        
        def get_child_tree(plan: QueryPlanRep):
            successor_tree = {} # Maps node in plan to successor
            for i in plan.ops_idx:
                for parent in plan.query_parent_map.get(i, []):
                    assert parent not in successor_tree, "One node should not have multiple parents in plan"
                    successor_tree[parent] = i
            return successor_tree

        def add_late_materialization(plan: QueryPlanRep, strategy: MaterialStrat):
            used_ids = set()
            for i in plan.ops_idx:
                for used_id in plan.ops_dict[i].get('output ids', []) + plan.ops_dict[i].get('input ids', []):
                    used_ids.add(used_id)

            if strategy == MaterialStrat.SIMPLE:
                for i in plan.ops_idx:
                    # print('-' * 30 + 'late materialization' + '-' * 30)
                    # print(plan.ops_dict[i])
                    if plan.ops_dict[i]['name'] == 'Project':
                        is_simple, material_set = is_simple_projection(plan.ops_dict[i]['output'])
                        # print(material_set)
                        if is_simple:
                            assert len(plan.query_parent_map[i]) == 1, "Multiple inputs into Project not anticipated"
                            parent = plan.query_parent_map[i][0]
                            if plan.ops_dict[parent]['name'] == 'Join':
                                plan.ops_dict[parent]['materialized in'] = list(material_set | set(plan.ops_dict[parent]['join in']))
                            plan.ops_dict[plan.query_parent_map[i][0]]['materialized out'] = list(material_set)
                            plan.ops_dict[i]['materialized out'] = []
                        # else:
                        #     ### TODO: Improve!!
                        #     plan.ops_dict[i]['materialized in'] = plan.ops_dict[i]['input ids'].copy()
                        #     plan.ops_dict[i]['materialized out'] = plan.ops_dict[i]['output ids'].copy()

                for i in plan.ops_idx:
                    #print(plan.ops_dict[i])
                    if plan.ops_dict[i]['name'] == 'Filter' and len(plan.ops_dict[i]['materialized out']) > 0:
                        mask_id = self.get_next_mask_id(used_ids)
                        plan.ops_dict[i]['materialized out'] = [mask_id] ## Now we only output mask for late materialization
                        child = plan.query_child_map[i]

                        while True:
                            if 'mask map' not in plan.ops_dict[child]:
                                plan.ops_dict[child]['mask map'] = {}
                            for j in plan.ops_dict[child]['input ids']:
                                if j in plan.ops_dict[i]['output ids']:
                                    plan.ops_dict[child]['mask map'][j] = mask_id
                            if plan.ops_dict[child]['name'] in ['Aggregate', 'Join', 'Sort', 'TakeOrderedAndProject']: ### We materialize here
                                break
                            child = plan.query_child_map[child]
                            
                        # plan.ops_dict[i]['output ids'].append(mask_id) 
                        plan.ops_dict[i]['out mask id'] = mask_id
                        # print(plan.ops_dict[i])
                        # plan.ops_dict[i]['material set'] = plan.ops_dict[i]['output ids'].copy()
                    # print(plan.ops_dict[i]['name'])
                    # print({key: plan.ops_dict[i][key] for key in ['input ids', 'output ids', 'material set'] if key in plan.ops_dict[i]})
            else:
                raise NotImplementedError("Not Implemented")
                
        def remove_exchange(plan: QueryPlanRep):
            marked_del = []
            for i in plan.ops_idx:
                if plan.ops_dict[i]['name'] == 'Exchange':
                    parent = plan.query_parent_map[i]
                    assert len(parent) == 1, "Exchange should only have 1 parent"
                    
                    child = plan.query_child_map[i]
                    plan.query_child_map[parent[0]] = child
                    ind = plan.query_parent_map[child].index(i)
                    plan.query_parent_map[child][ind] = parent[0]
                    marked_del.append(i)
            
            for d in marked_del:
                plan.ops_idx.remove(d)
                plan.query_child_map.pop(d)
                plan.query_parent_map.pop(d)
                plan.ops_dict.pop(d)
                        
        def rewrite_ops(plan: QueryPlanRep):
            # Rewrite join based on build side
            for i in plan.ops_idx:
                if plan.ops_dict[i]['name'] == 'Join':
                    assert len(plan.query_parent_map[i]) == 2, "Plan should accept 2 sets of inputs"
                    ls, rs = plan.query_parent_map[i]
                    left_keys, right_keys = plan.ops_dict[i]['left keys'], plan.ops_dict[i]['right keys']
                    if plan.ops_dict[i]['build'] == 'right':
                        plan.ops_dict[i]['right keys'], plan.ops_dict[i]['left keys'] = left_keys, right_keys
                        ls, rs = rs, ls
                    else:
                        assert 'Left' not in plan.ops_dict[i]['join type'], "Anti/Semi/Outer only support BuildRight"
    
                    plan.ops_dict[i]['left group'] = plan.ops_dict[ls]['output ids']
                    plan.ops_dict[i]['right group'] = plan.ops_dict[rs]['output ids']
                    plan.ops_dict[i]['input ids'] = plan.ops_dict[i]['left group'] + plan.ops_dict[i]['right group']
                    plan.ops_dict[i]['output ids'] = plan.ops_dict[i]['input ids'].copy()

                    used_set = set()
                    for key in plan.ops_dict[i]['left keys'] + plan.ops_dict[i]['right keys']:
                        used_set.add(key)
                    if 'join condition' in plan.ops_dict[i]:
                        get_literals(plan.ops_dict[i]['join condition'], used_set)
                        
                    plan.ops_dict[i]['join in'] = list(used_set)
                    plan.ops_dict[i]['materialized out'] = plan.ops_dict[i]['output ids'].copy()
                    plan.ops_dict[i]['materialized in'] = plan.ops_dict[i]['materialized out'].copy()

                elif plan.ops_dict[i]['name'] == 'Filter':
                    used_set = set()
                    get_literals(plan.ops_dict[i]['filter tree'], used_set)
                    used_set = used_set & set(plan.ops_dict[i]['input ids'])
                    plan.ops_dict[i]['materialized in'] = list(used_set)
                    plan.ops_dict[i]['materialized out'] = list(used_set)
                
                elif plan.ops_dict[i]['name'] == 'Aggregation':
                    plan.ops_dict[i]['materialized in'] = plan.ops_dict[i]['input ids'].copy()
                    plan.ops_dict[i]['materialized out'] = plan.ops_dict[i]['output ids'].copy()
                
                elif plan.ops_dict[i]['name'] == 'Project':
                    rename_map, material_in, material_out, used_expr = get_project_used_cols(plan.ops_dict[i]['output'], set(plan.ops_dict[i]['input ids']))
                    plan.ops_dict[i]['rename map'] = rename_map
                    plan.ops_dict[i]['materialized in'] = list(material_in)
                    plan.ops_dict[i]['materialized out'] = material_out
                    plan.ops_dict[i]['used expr'] = used_expr


        
        def clean_agg(plan: QueryPlanRep):
            ## Deal with duplicate aggregation and deal with Q16 special case
            marked_del = []
            if qid == 16:
                marked_del.extend([13, 14, 15]) ## Issue with Spark Q16 Plan generation, skip 13-15
                plan.query_parent_map[16] = [12]

            for i in plan.ops_idx:
                if qid == 16 and i in [13, 14, 15]:
                    continue
                if plan.ops_dict[i]['name'] == 'Aggregate' and plan.ops_dict[i]['is partial']:
                    assert plan.ops_dict[i+1]['name'] == 'Exchange' and plan.ops_dict[i+2]['name'] == 'Aggregate'
                    plan.ops_dict[i+2]['input ids'] = plan.ops_dict[i]['input ids'] # Only need to swap input ids
                    plan.query_parent_map[i+2] = plan.query_parent_map[i]
                    marked_del.extend([i+1, i])

            for d in marked_del:
                plan.ops_idx.remove(d)
                plan.query_parent_map.pop(d)
                plan.ops_dict.pop(d)

        def extract_input_output(plan: QueryPlanRep):
            ## Template here
            for i in plan.ops_idx:
                if plan.ops_dict[i]['name'] == 'Scan':
                    plan.in_col_dict[i] = []
                    plan.out_col_dict[i] = plan.ops_dict[i]['output ids']
                elif plan.ops_dict[i]['name'] == 'Filter':
                    plan.in_col_dict[i] = plan.ops_dict[i]['input ids']
                    plan.out_col_dict[i] = plan.ops_dict[i]['output ids']
                elif plan.ops_dict[i]['name'] == 'Project':
                    plan.in_col_dict[i] = []
                    plan.out_col_dict[i] = []
                elif plan.ops_dict[i]['name'] == 'Aggregate':
                    plan.in_col_dict[i] = []
                    plan.out_col_dict[i] = []
                elif plan.ops_dict[i]['name'] == 'Join':
                    plan.in_col_dict[i] = []
                    plan.out_col_dict[i] = []
                elif plan.ops_dict[i]['name'] == 'Sort':
                    plan.in_col_dict[i] = []
                    plan.out_col_dict[i] = []
                elif plan.ops_dict[i]['name'] == 'Exchange':
                    plan.in_col_dict[i] = []
                    plan.out_col_dict[i] = []
                elif plan.ops_dict[i]['name'] == 'take ordered':
                    plan.in_col_dict[i] = []
                    plan.out_col_dict[i] = []
                else:
                    plan.in_col_dict[i] = []
                    plan.out_col_dict[i] = []
        
        parsed = []
        for trees, lines in zip([sub_tree_lines, tree_lines], [sub_item_lines, item_lines]):
            if lines is not None:
                plan = QueryPlanRep()
                plan.ops_idx, plan.ops_dict = parse_item(lines)
                parse_tree(trees, plan.query_parent_map, plan.ops_dict)
                clean_agg(plan)
                plan.query_child_map = get_child_tree(plan)
                remove_exchange(plan)
                rewrite_ops(plan)
                add_late_materialization(plan, MaterialStrat.SIMPLE)
                parsed.append(plan)
        return parsed

        
