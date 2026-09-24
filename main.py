from tool import TOOLS
from ml_agent.reasoning import REASONING

def run_agent(file_obj, target):
    # Read dataset using TOOLS
        df = TOOLS["input_file"](file_obj)
    
        if df is None:
            return None
    
        # Check target using TOOLS
        target_type = TOOLS["target_check"](df, target)
    
    
        if target_type is None:
            return None

        # Reasoning if some column isn't needed
        result1=REASONING["data_manage"](df,target)

        # HITL for column deletion using TOOL
        df=TOOLS["hitl_del"](df,result1)

        # Varifying column data type using REASONING
        result2=REASONING["columns_check"](df)

        # HITL for changing column data type using TOOLS
        df= TOOLS["hitl_colume_change"](df,result2)

        # EDA tool
        original_null_columns,total_columns,numerical_columns,categorical_columns = TOOLS["eda"](df) # Get the list of null columns
    
        # HITL NULL tools
        df, remain_col = TOOLS["hitl_null"](df, original_null_columns)  # Get the DataFrame after handling null values
        null_columns = len(remain_col)  # Get the number of null columns after handling

        if null_columns!=0:
            # again EDA after NULL handling
            null_columns,total_columns,numerical_columns,categorical_columns = TOOLS["eda"](df)

        # Decide the algorithm
        result3 = REASONING["choose_boosting_algorithm"](df,target,target_type,total_columns,numerical_columns,categorical_columns,null_columns)
    
        return result3

if __name__ == "__main__": 
    file_name=input("Type the address of dataset file: ")
    df = TOOLS["input_file"](file_name) 

    target = input("Type the target variable: ")

    result = run_agent(file_name, target)
    if result is not None:

        Best_algorithm = result["Best algorithm"]
        print("=============================")
        print("Recommended algorithm :",Best_algorithm)
        Reason_of_best_algorithm=result["Reason_of_best_algorithm"]
        print("Reason it is recommended :",Reason_of_best_algorithm)

        print("============================")
        second_best_algorithm=result["2nd_best_algorithm"]
        print("Alternative algorithm :",second_best_algorithm)
        Reason_of_2nd_best_algorithm=result["Reason_of_2nd_best_algorithm"]
        print("Reason it's 2nd most suitable :",Reason_of_2nd_best_algorithm)

        print("============================")
        Remark=result["Remark_for_further_Ml_processing"]
        print("Remark for further ML processing :",Remark)