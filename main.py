import lock_in_data
import pandas as pd
import pickle
from scipy.constants import pi as pi
import numpy as np
import warnings


warnings.filterwarnings(action='ignore')


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
        print_param=False
    )
    
    phaseshifter_phase_fig.savefig(f"./results/phaseshifter_phase_freq_plot(phase{phase}).png")

experiment = 'Low-pass Amplifier'
print(experiment)

def lpf_fitting(
    log_f,tau
):
    f= 10**log_f
    return -10*np.log10(1+(2*pi*f*tau)**2)


for db_oct in [6,12]:
    for time_constant in [0.03,0.1,0.3]:


        Low_pass_amplifier_gain_fig = lock_in_data.phys_plot(
            datum[experiment],
            'none',
            lambda x : 20*np.log10(x.results[3]/x.results[2]),
            {'dB/oct' : db_oct, 'Time Constant' : time_constant},
            x_label = "log(frequency)",
            y_label = "$\Delta$dB",
            fmt = "ko",
            x_variable_function = lambda x: np.log10(float(x.parameter['frequency(log scale)'][0:-2])) if x.parameter['frequency(log scale)'][-3]!='m' else np.log10(float(x.parameter['frequency(log scale)'][0:-3])/1000),
            additional_line = -3,
            fitting_function=lpf_fitting,
            p0=[time_constant],
            error_bar_y = lambda x: 2 if x<0.5 else 5,
            print_param=True,
            dosing_y=-3,
            dosing_x0=0.5
        )

        
        Low_pass_amplifier_gain_fig.savefig(f"./results/low_pass_filter_gain_freq_plot(db_oct{db_oct})(time_constant{time_constant}).png")



# for data in datum[experiment]:
#     data.print_data()
#     data.full_image.save("./ERROR.PNG")
#     a = input("Commit data:")
#     try:
#         data.results[3]=float(a)
#         print("Modified")
#     except ValueError:
#         continue




# with open("./datum.pkl", "wb") as f:
#     pickle.dump(datum,f)
