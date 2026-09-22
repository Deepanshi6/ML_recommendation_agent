import pandas as pd
from typing import IO


def input_file(file_obj: IO[str]):
    try:
        df = pd.read_csv(file_obj)
        print("Valid CSV")
        return df
    except Exception as e:
        print("Not a valid CSV")
        print(e)
        return None

def target_check(df,target):
    try:
        target_type=df[target].dtypes
        if target_type=='object' or target_type=='category' or target_type=='bool':
            print('target variable type : ',df[target].dtypes)
            return 'categorical'
        elif target_type=='int64' or target_type=='float64':
            print('target variable type : ',df[target].dtypes)
            return 'numerical'
    except:
        print('Invalid target variable.')

def hitl_del(df, result1):

    print("Columns that require action :")

    columns = result1.get("columns", [])

    if not columns:
        print("No such column.")
        return df

    for item in columns:

        column = item["column"]
        reason = item["reason"]

        print("\nColumn :", column)
        print("Reason :", reason)

        while True:
            act = input("Do you want to remove the column (y/n) : ").strip().lower()

            if act == "y":
                df = df.drop(columns=[column])
                print(f"Removed: {column}")
                break

            elif act == "n":
                print(f"Kept: {column}")
                break

            else:
                print("Please enter y or n.")

    return df


def hitl_colume_change(df,result2):
    print("Column for which the datatype should be changed :")

    if len(result2["columns"])==0:
        print("No such column")
        return df
    else:
        for i in result2["columns"]:
            print("Column Name :",i["column"])
            print("Detected data type :",i["detected_type"])
            print("Suggested data type :",i["suggested_dtype"])

            while True:

                act=input("Do you want to change to suggested data type (y/n) :")
                if act=="y":
                    df[i["column"]]=df[i["column"]].astype(i["suggested_dtype"])
                    print("Changed the data type.")
                    break
                elif act == "n":
                    print("no change in data type done.")
                    break
        return df

def eda(df):
    # Total number of columns
    total_columns = df.shape[1]
    print("Number of columns:", total_columns)

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
        ).columns.tolist()

    numerical_columns = df.select_dtypes(
        include=["number"]
        ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime", "datetimetz"]
        ).columns.tolist()

    print("Number of categorical columns:", len(categorical_columns))
    print("Number of numerical columns:", len(numerical_columns))
    print("Number of date/datetime columns:", len(datetime_columns))

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nDate/Datetime columns:")
    print(datetime_columns)
    columns = df.columns.tolist()
    null_status = False
    n_t = []
    nt_t=[]
    nn_t=[]
    ne_t=[]
    for column_name in columns:
        column = df[column_name]
        total_values = len(column)
        null_values = column.isna().sum()
        empty_values = column.astype("string").str.strip().eq("").fillna(False).sum()

        if null_values + empty_values >= total_values / 10:
            null_status = True
            n_t.append(column_name)
            nt_t.append(total_values)
            nn_t.append(null_values)
            ne_t.append(empty_values)
    print("----------------------------------------")
    print("Null status:", null_status)
    if null_status:
        print("Null status columns:")
        for column_name in n_t:
            print(column_name)
            print("total values :",nt_t)
            print("null values :",nn_t)
            print("empty values :",ne_t)
    return n_t,total_columns,categorical_columns,numerical_columns

def hitl_null(df,n_t):
    print("-----------------------------")
    null_status=False
    remain_col=[]
    for i in n_t:
        print("What shall be done with null values for the column :",i)
        print("1 Drop the null columns")
        print("2 Drop the null rows from whole data")
        print("3 Fill the null values with mode")
        print("4 Fill null values with mean")
        print("5 Consider the null values for further model training.")
        a = int(input("Type a number (1/2/3/4/5): "))
        if a==1:
            df.drop(columns=[i], inplace=True)
            print("Dropped the column :",i)
        elif a==2:
            df.dropna(subset=[i], inplace=True)
            print("Dropped the null rows.")
        elif a==3:
            df[i]=df[i].fillna(df[i].mode()[0])
            print("Filled the null values with mode")
        elif a==4:
            df[i]=df[i].fillna(df[i].mean())
            print("Filled the null values with mean")
        elif a==5:
            null_status=True
            remain_col.append(i)
            print("Column considered for further processing.")
    return df,remain_col

TOOLS={
    "input_file":input_file,
    "target_check":target_check,
    "hitl_del":hitl_del,
    "hitl_colume_change":hitl_colume_change,
    "eda":eda,
    "hitl_null":hitl_null
}