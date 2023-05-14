import pytesseract as pyt
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import numpy as np

exceptional_cases = [528,529,530,531,532,533,534]
#fft data are excepted since readinng method cannot reach.
exceptional_data_label = ["fft_freq_a", "fft_results_a", "fft_freq_b", "fft_results_b"]

class lock_in_data:
    file_name : str
    full_image : Image
    cropped_images : Image
    data_label : list
    results : list
    parameter : dict
    tessaract_mode : bool

    def __init__(self, file_name, data_label, parameter, tessaract_mode = False):
        self.file_name = file_name
        self.data_label = data_label
        self.parameter = parameter
        self.full_image = Image.open(self.file_name)
        self.tessaract_mode = tessaract_mode
        self.read_png()

    def read_png(self):

        self.results = []
        if self.data_label == exceptional_data_label:
            print("Exceptional read_png occurs. Please read FFT data from ERROR.PNG.")
            print("Write data in order of", self.data_label , "in unit of Hz and dB")
            self.full_image.save("./ERROR.PNG")
            for i in range(4):
                a = input("Commit data : ")
                self.results.append(a)
            
            return

        self.cropped_images = []
        if len(self.data_label) == 4:
            crop_tuples = [(240,206,300,219),(360,206,420,219),(240,218,300,232),(360,218,418,232)]
        if len(self.data_label) == 3:
            crop_tuples = [(240,206,300,219),(360,206,420,219),(240,218,300,232)]
        if len(self.data_label) == 2:
            crop_tuples = [(240,220,300,232),(360,220,418,232)]
        if len(self.data_label) == 1:
            crop_tuples = [(240,220,300,232)]

        for crop_tuple in crop_tuples:
            self.cropped_images.append(self.full_image.crop(crop_tuple))
        for cropped_image, data_type in zip(self.cropped_images, range(len(self.data_label))):

            if self.tessaract_mode == False:
                cropped_image.save("./ERROR.PNG")
                result = input("Commit data :")
            else:
                result_string = pyt.image_to_string(cropped_image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.-')
                try:
                    result=float(result_string)
                    if 'phase'  in self.data_label[data_type]:
                        if  np.abs(result) > 180 or result_string == ' ':
                            print("Inaccurate reading may occurs. Please check and read ERROR.PNG" )
                            cropped_image.save("./ERROR.PNG")
                            result = input("Commit data : ")

                except ValueError:
                    print("Unreadable error occurs. Please read ERROR.PNG")
                    cropped_image.save("./ERROR.PNG")
                    result = input("Commit data : ")
                
                
            self.results.append(result)

    def print_data(self):
        for data_type in range(len(self.data_label)):
            print(self.data_label[data_type], ':' , self.results[data_type])
        print(self.parameter)
                


def clean_dataframe(
    df
) :
    colomn_dict = df.columns

    for colomn in colomn_dict:
        if colomn == 'name':
            break

        for index in range(df.shape[0]):
            if type(df.loc[index,colomn]) == str:
                key=df.loc[index,colomn]
                continue
            if np.isnan(df.loc[index,colomn]) == False:
                key=df.loc[index,colomn]
                continue
            df.loc[index,colomn] = key

    return df


def get_data_labels(
    experiment
):
    if experiment == 'preamplifier':
        return ["1_amplitude","2_amplitude","12_phase","21_phase"]
    elif experiment == 'phaseshifter':
        return ["1_amplitude", "2_amplitude", "12_phase", "21_phase"]
    elif experiment == 'DBM' :
        return ["1_amplitude","31_phase", "3_frequency", "2_area"]
    elif experiment == 'Low-pass Amplifier':
        return ["1_frequency", "2_frequency", "1_amplitude", "2_amplitude"]
    elif experiment == 'Lock-In Amplifier(Noise+signal)':
        return ["1_frequency","2_frequency","1_amplitude","2_amplitude"]
    elif experiment == "Lock-In Amplifier(Signal 검출)":
        return ["12_phase","3_area"]
    elif experiment == "Lock-In Amplifier(DC offset)":
        return ["12_phase", "3_area"]
    elif experiment == "Hall Effect":
        return ["1_frequency", "12_phase", "3_area"]
    else:
        print("No experiment name" , experiment)
        return NameError


def phys_plot(
    data_set,
    x_parameter,
    dep_variable_function,
    fixed_parameter,
    x_label,
    y_label,
    fmt,
    x_variable_function='none',
    fitting_function = 'none',
    p0=0,
    additional_line = 'none',
    error_bar_y = 'none',
    print_param= False,
    dosing_y= 'none',
    dosing_x0 = 'none'
): 
    #data set : list of lock_in_data type
    #x_parameter : the x-variable of a figure, must be a key of an data parameter dictionary
    #dep_variable_function : the y-variable of a figure, must be a lambda function of an data results.
    #fixed_parameter: the parameter which should be fixed. It must be a type of subdictionary of a datatype
    #x_variable_function : besides x_parameter, can use x_variable_function option.
    #fitting_function : fitting function of the plot
    #p0 : initial parameter setting
    #additional line : additional line setup attribute
    #error bar mode : input of lambda function of stderr in data x.
    #print_param : to print regression results.\
    #dosing_y : find x value when fitting function results = dosing_y

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)

    x_list=[]
    y_list=[]



    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for data in data_set:
        if dictionary_boolean(data.parameter,fixed_parameter):
            if  x_variable_function == 'none':
                x_list.append(data.parameter[x_parameter])
            else:
                x_list.append(x_variable_function(data))

            y_list.append(dep_variable_function(data))

    x= np.array(x_list)
    y= np.array(y_list)

    x_con=np.linspace(min(x_list),max(x_list),50)
    if fitting_function != 'none':
        param,_=get_regression_result(x_list,y_list,fitting_function,p0)
        ax.plot(x_con,fitting_function(x_con,*param),'r-')
        if print_param == True:
            print(param)
            residuals = y - fitting_function(x,*param)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y-np.mean(y))**2)
            print("R^2 = " ,1-ss_res/ss_tot)

        if dosing_y != 'none':
            print("dosing results: ",opt.fsolve(lambda x: fitting_function(x,*param)-dosing_y,dosing_x0))



    if additional_line != 'none':
        y_con = np.ones_like(x_con,dtype=np.float64)
        y_con= y_con*additional_line
        ax.plot(x_con,y_con,'b-')

    

    if error_bar_y != 'none':
        yerr = [error_bar_y(x_element) for x_element in x]
        plt.errorbar(x,y,yerr=yerr,fmt=fmt)
    else:
        ax.plot(x,y,fmt)
    

    fig.tight_layout()
    return fig



def dictionary_boolean(
    dic1, dic2
):
    if len(dic1) < len(dic2):
        dic2, dic1 = zip(dic1,dic2)

    
    for key in list(dic2.keys()):
        try:
            if dic1[key] != dic2[key]:
                return False
        except KeyError:
            print("Unexpected key from dict boolean")
            return KeyError
    
    return True


def get_regression_result(
    x,y,fitting_function, p0
):
    param, param_covariance=opt.curve_fit(
        fitting_function,
        x,
        y,
        p0,
        maxfev=60000
    )

    return param,param_covariance
