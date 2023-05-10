import lock_in_data
import pandas as pd
import pickle


index_label_file_name = "./LI_picturename.xlsx"
df = pd.read_excel(index_label_file_name, sheet_name= None)
experiments = df.keys()
datum = {experiment : [] for experiment in experiments}

num_pic = 701

try:
    with open("datum.pkl","rb") as f:
        datum = pickle.load(f)

except FileNotFoundError:
    parameters = [{} for i in range(num_pic)]
    data_labels = [{} for i in range(num_pic)]
    experiment_labels = ["" for i in range(num_pic)]

    for experiment in experiments:
        dataframe = lock_in_data.clean_dataframe(df[experiment])
        column_dict = dataframe.columns
        # clean up filled NaNs

        for index in range(dataframe.shape[0]): #repeatly read for column numbers
            parameter = {}
            for column in column_dict:
                if column == 'name':
                    break
                parameter[column]=dataframe.loc[index,column]
            
            pic_idx= dataframe.loc[index,'name']

            if pic_idx >= num_pic:
                break
            
            parameters[pic_idx]=parameter
            experiment_labels[pic_idx] = experiment
            if pic_idx in lock_in_data.exceptional_cases:
                data_labels[pic_idx]=lock_in_data.exceptional_data_label
                continue

            data_labels[pic_idx]=lock_in_data.get_data_labels(experiment)


    for index in range(0,num_pic):
        if index == 350 : #No files. very personal.
            continue
        print(f"reading {index} out of {num_pic}")
        datum[experiment_labels[index]].append(lock_in_data.lock_in_data(file_name = f"./raw_data/TEK00{format(index, '03')}.PNG", data_label = data_labels[index] , parameter=parameters[index]))
        datum[experiment_labels[index]][-1].print_data()

    with open("./datum.pkl", "wb") as f:
        pickle.dump(datum,f)


experiment = 'preamplifier'
fig = lock_in_data.phys_plot(datum[experiment],'frequency(kHz)', lambda x : x.results[0]/x.results[1], {'gain' : 1},x_label="frequency [kHz]" , y_label= "Amplitude_ratio", fmt = 'ko')
fig.savefig(f"./results/preamplifier_amp_ratio_freq_plot(gain{1}).png")


# for experiment in experiments:
#     print(f"For {experiment},")       
#     for data in datum[experiment]:
#         data.print_data()
        

