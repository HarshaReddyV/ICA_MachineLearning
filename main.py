from utils import BackFillData, VisualiseData

def main():
    print('Starting Programme')
    print('Starting processing data')
    BackFillData()
    print('Finished Processing Data and outputted the data to FTSE100_DataProcessed')
    print('Start Generating initial data visualisation')
    VisualiseData()
    print('Finished Initial  visuals')





if __name__ == "__main__":
    main()
