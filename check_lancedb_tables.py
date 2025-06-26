import lancedb
import os

db_path = "./rag_system/index_store/lancedb"

if not os.path.exists(db_path):
    print(f"LanceDB directory not found at: {db_path}")
    exit()

try:
    db = lancedb.connect(db_path)
    table_names = db.table_names()

    print("--- Existing LanceDB Tables ---")
    if table_names:
        for name in table_names:
            print(f"- {name}")
    else:
        print("No tables found in LanceDB.")
    print("-----------------------------")

except Exception as e:
    print(f"An error occurred: {e}") 