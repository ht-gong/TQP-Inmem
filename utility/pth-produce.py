import os
import pandas as pd
import numpy as np
import torch
import json

os.chdir('/work1/talati/zhaoyah/TQP-Vortex')

with open('config.json', 'r') as f:
    config = json.load(f)

epoch = pd.Timestamp('1990-01-01')

lineitem_columns = [
          "L_ORDERKEY",         # identifier
           "L_PARTKEY",         # identifier
           "L_SUPPKEY",         # identifier
           "L_LINENUMBER",      # integer (x)
           "L_QUANTITY",        # decimal !
           "L_EXTENDEDPRICE",   # decimal !
           "L_DISCOUNT",        # decimal !
           "L_TAX",             # decimal !
           "L_RETURNFLAG",      # fixed text, size 1 !
           "L_LINESTATUS",      # fixed text, size 1 !
           "L_SHIPDATE",        # date !
           "L_COMMITDATE",      # date
           "L_RECEIPTDATE",     # date
           "L_SHIPINSTRUCT",    # fixed text, size 25 !
           "L_SHIPMODE",        # fixed text, size 10
           "L_COMMENT"          # variable text size 44 !
           ]
lineitem_types = [int, int, int, int, float, float, float, float,
         1, 1, 'date', 'date', 'date', 25, 10, 44]
lineitem_used = ["L_ORDERKEY", "L_PARTKEY", "L_SUPPKEY", "L_COMMITDATE",
        "L_RECEIPTDATE", "L_SHIPMODE"]

# part
part_columns = [
    "P_PARTKEY",        # identifier
    "P_NAME",           # variable text, size 55
    "P_MFGR",           # fixed text, size 25
    "P_BRAND",          # fixed text, size 10
    "P_TYPE",           # variable text, size 25
    "P_SIZE",           # integer
    "P_CONTAINER",      # fixed text, size 10
    "P_RETAILPRICE",    # decimal
    "P_COMMENT"         # variable text, size 23
]
part_types = [int, 55, 25, 10, 25, int, 10, float, 23]

# supplier
supplier_columns = [
    "S_SUPPKEY",        # identifier
    "S_NAME",           # fixed text, size 25
    "S_ADDRESS",        # variable text, size 40
    "S_NATIONKEY",      # identifier
    "S_PHONE",          # fixed text, size 15
    "S_ACCTBAL",        # decimal
    "S_COMMENT"         # variable text, size 101
]
supplier_types = [int, 25, 40, int, 15, float, 101]

# partsupp
partsupp_columns = [
    "PS_PARTKEY",       # identifier
    "PS_SUPPKEY",       # identifier
    "PS_AVAILQTY",      # integer
    "PS_SUPPLYCOST",    # decimal
    "PS_COMMENT"        # variable text, size 199
]
partsupp_types = [int, int, int, float, 199]

# customer
customer_columns = [
    "C_CUSTKEY",        # identifier
    "C_NAME",           # fixed text, size 25
    "C_ADDRESS",        # variable text, size 40
    "C_NATIONKEY",      # identifier
    "C_PHONE",          # fixed text, size 15
    "C_ACCTBAL",        # decimal
    "C_MKTSEGMENT",     # fixed text, size 10
    "C_COMMENT"         # variable text, size 117
]
customer_types = [int, 25, 40, int, 15, float, 10, 117]

# orders
orders_columns = [
    "O_ORDERKEY",       # identifier
    "O_CUSTKEY",        # identifier
    "O_ORDERSTATUS",    # fixed text, size 1
    "O_TOTALPRICE",     # decimal
    "O_ORDERDATE",      # date
    "O_ORDERPRIORITY",  # fixed text, size 15
    "O_CLERK",          # fixed text, size 15
    "O_SHIPPRIORITY",   # integer
    "O_COMMENT"         # variable text, size 79
]
orders_types = [int, int, 1, float, 'date', 15, 15, int, 79]

# nation
nation_columns = [
    "N_NATIONKEY",      # identifier
    "N_NAME",           # fixed text, size 25
    "N_REGIONKEY",      # identifier
    "N_COMMENT"         # variable text, size 152
]
nation_types = [int, 25, int, 152]

# region
region_columns = [
    "R_REGIONKEY",      # identifier
    "R_NAME",           # fixed text, size 25
    "R_COMMENT"         # variable text, size 152
]
region_types = [int, 25, 152]

def produce(SF, name, columns, types, used):
    
    print(f"SF = {SF}")
    file_path = os.path.join(config.get('tables path'), f'{name}-{SF}.tbl')
    print(f"Reading {file_path}")

    df = pd.read_csv(file_path, sep=',', header=None, dtype=str)
    df.columns = columns
    print(df.head())

    def str_to_tensor(s, length):
        encoded = torch.tensor([ord(c) for c in s[:length]], dtype=torch.int8)
        padded = torch.nn.functional.pad(encoded, (0, length - len(encoded)))
        return padded
    
    flag = False
    for col, ty in zip(columns, types):

        if col.upper().endswith('COMMENT') and col[0] not in ['S', 'C', 'O']:
            continue
        
        if col not in used:
            continue
        
        print (f"Processing [{col}]")
        if isinstance(ty, int):
            lim = ty
            if lim == 1:
                tensor_data = torch.tensor(df[col].apply(lambda s: ord(s)).values, dtype=torch.int8)
            else:
                tensor_col = []
                for i in range(lim):
                    print (f"Ch {i+1}/{lim}")
                    tensor_col.append(torch.tensor(df[col].apply(lambda s: (ord(s[i]) if i < len(s) else 0)).values, dtype=torch.int8).unsqueeze(1))

                tensor_data = torch.cat(tensor_col, dim=1)
        elif ty == int:
            tensor_data = df[col].astype(int).values
        elif ty == float:
            tensor_data = df[col].astype(float).values
        elif ty == 'date':
            df[col] = pd.to_datetime(df[col])
            tensor_data = ((df[col] - epoch).dt.total_seconds() * 1e9).astype(int).values
        
        tensor = torch.tensor(tensor_data)
        print (f"Saving [{col}] (ROWS = {tensor.shape})")
        tensor_path = os.path.join(config.get('tensors path'), f'SF{SF}-tensor-{col}.pth')
        torch.save(tensor, tensor_path)
        del tensor, tensor_data

torches = {}

SF = 500

# produce(SF, "nation", nation_columns, nation_types, [])
# produce(SF, "region", region_columns, region_types, [])
# produce(SF, "supplier", supplier_columns, supplier_types, [])
# produce(SF, "customer", customer_columns, customer_types, [])
# produce(SF, "part", part_columns, part_types, [])

# produce(SF, "orders", orders_columns, orders_types, [])
# produce(SF,  "partsupp", partsupp_columns, partsupp_types, [])
produce(SF, "lineitem", lineitem_columns, lineitem_types, lineitem_used)