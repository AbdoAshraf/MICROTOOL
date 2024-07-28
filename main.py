from database_analysis import DatabaseAnalysis
from source_code_analysis import SourceCodeAnalysis

def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Windwos.2000',
        'database': 'day_trader'
    }

    # API key for OpenAI
    api_key = "apikey"

    # Paths to files
    descriptions_file = 'C:\\Masters_Tool\\MicroTool\\MicroTool\\table_descriptions_day_trder_accurate.json'
    java_directory = 'C:\\Masters_Tool\\lastTrial\\call-graph-parser\\daytrader\\sample.daytrader7-master'
    excel_filepath = 'C:\\Masters_Tool\\lastTrial\\call-graph-parser\\output.xlsx.xlsx' # Paths to files
    
    # descriptions_file = 'C:\\Masters_Tool\\MicroTool\\MicroTool\\table_descriptions_jpetstore.json'
    # java_directory = 'C:\\Masters_Tool\\jpetstore-6-master\\jpetstore-6-master\\src\\main\\java'
    # excel_filepath = 'C:\\Masters_Tool\lastTrial\\call-graph-parser\\JpetStore\\jpetStoreCallgraph.xlsx'
    # bcs_per_class_filepath = 'C:\\Masters_Tool\\MicroTool\\MicroTool\\day_trader_bcs_per_class.json'

    # Initialize and run the database analysis
    db_analysis = DatabaseAnalysis(db_config, api_key, descriptions_file)
    db_analysis.start_database_analysis()
    db_clusters = db_analysis.db_clusters

    # Initialize and run the source code analysis
    source_code_analysis = SourceCodeAnalysis(java_directory, excel_filepath, db_clusters)
    source_code_analysis.start_source_code()

if __name__ == "__main__":
    main()
