import os
import pandas as pd



class HomeCredit:

    def get_data(self):
        """
        This function returns a Python dict.
        Its values should be pandas.DataFrame loaded from csv files
        """
        # Hints: Build csv_path as "absolute path" in order to call this method from anywhere.
       
        # Use __file__ as absolute path anchor independant of your computer
        # Make extensive use of `import ipdb; ipdb.set_trace()` to investigate what `__file__` variable is really
        # Use os.path library to construct path independent of Unix vs. Windows specificities
        pathh = os.path.dirname(os.getcwd())
        csv_path = os.path.join(pathh, "raw_data")
        #print(pathh)
        #print(csv_path)
        ##
        file_names = []
        for i in os.listdir(csv_path):
            if (".csv" in i):
                file_names.append(i)
        ##
        key_names = [i.replace(".csv", "").replace("application_", "") for i in file_names]
        #
        data_keys = {}
        for (x, y) in zip( key_names, file_names):
            data_keys[x] = y
        data_keys
        
        data = {}
        for k,l in data_keys.items():
            data[k] =  pd.read_csv(os.path.join(csv_path, l))
        #dataframe = pd.read_csv(os.path.join(csv_path, y)).head())
        return data
