import lock_in_data
import pandas as pd
import pickle
from scipy.constants import pi as pi
import numpy as np

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

print(experiment)
for gain in [1,2,5,10,20]:
    preamplifier_gain_fig = lock_in_data.phys_plot(
        datum[experiment],
        'frequency(kHz)',
        lambda x : x.results[0]/x.results[1],
        {'gain' : gain},
        x_label="frequency [kHz]" ,
        y_label= "gain(output/input)",
        fmt = 'ko',
        additional_line = gain
        )
    
    preamplifier_gain_fig.savefig(f"./results/preamplifier_gain_freq_plot(gain{gain}).png")

    preamplifier_phase_fig = lock_in_data.phys_plot(
        datum[experiment],
        'frequency(kHz)',
        lambda x: x.results[2] if x.results[2]>0 else x.results[2]+360,
        {'gain' : gain},
        x_label="frequency [kHz]" ,
        y_label= "phase shift [$\degree$]",
        fmt = 'ko',
    )
    
    preamplifier_phase_fig.savefig(f"./results/preamplifier_phase_freq_plot(gain{gain}).png")


experiment = 'phaseshifter'

print(experiment)
for phase, Delta in zip([0,90,180,270],[150,50,-200,-300]):
    #Delta is very practical value of 2pi shifting

    phaseshifter_gain_fig = lock_in_data.phys_plot(
        datum[experiment],
        'frequency(Hz)',
        lambda x : x.results[0]/x.results[1],
        {'phase' : phase, 'fine phase' : 0},
        x_label="frequency [Hz]" ,
        y_label= "gain(output/input)",
        fmt = 'ko',
        )
    
    phaseshifter_gain_fig.savefig(f"./results/phaseshifter_gain_freq_plot(phase{phase}).png")

    phaseshifter_phase_fig = lock_in_data.phys_plot(
        datum[experiment],
        'frequency(Hz)',
        lambda x: x.results[2]-phase if x.results[2]-phase>Delta else x.results[2]-phase+360,
        {'phase' : phase, 'fine phase' : 0},
        x_label="frequency [Hz]" ,
        y_label= "phase shift [$\degree$]",
        fmt = 'ko',
        fitting_function=lambda x,a,b : a*x+b,
        p0=[3.75,0],
        error_bar_y= lambda x: 20,
        print_param=True
    )
    
    phaseshifter_phase_fig.savefig(f"./results/phaseshifter_phase_freq_plot(phase{phase}).png")

# for data in datum[experiment]:
#     data.print_data()
