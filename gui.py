
# Import the tkinter module
import tkinter
from tkinter import filedialog
class ui:
    def __init__(self):
        pass
    def maingui():
        # Create the default window
        root = tkinter.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.title("Decomposition")
        root.geometry(str(screen_width)+'x'+str(screen_height))
        command=0
          
        # Create the list of options
        # options_list1 = ["Krogager","Cameron","H/A/Alpha", "Huynen", "Barnes 1" ,"Barnes 2","Cloude",
        #                  "Unified Huynen","Holm 1","Holm 2", "An & Yang 3", "An & Yang 4", 
        #                  "Bhattacharya & Frery 4", "Freeman 2", "Freeman 3", "Neumann 2", "Arii 3 NNED", 
        #                  "Van Zyl 3", "Singh 4", "Yamaguchi 3", "Yamaguchi 4", "L Zhang 5", "Singh-Yamaguchi 6",
        #                  "Touzi", "Aghababaee","DI","DP","SD","SP","CC","cor_coef"]
        
        options_list1 = ["depol_index","deg_of_purity","scatt_div","scatt_pre","conf_coeff","corr_coef"]
        
        options_list2 = ["300","500","700","900","1100"]
        
        options_list3 = ["3", "5", "7", "9"]
        
        options_list4= {'Ascending Pass Data 1':'AP1','Ascending Pass Data 2':'AP2',
                        'Descending Pass Data 1':'DP1','Descending Pass Data 2':'DP2'}
          
        # Variable to keep track of the option
        # selected in OptionMenu
        value_inside1 = tkinter.StringVar(root)
        value_inside11 = tkinter.StringVar(root)
        value_inside111 = tkinter.StringVar(root)
          
        # Set the default value of the variable
        value_inside1.set("Select parameter1")
        value_inside11.set("Select parameter2")
        value_inside111.set("Select parameter3")
          
        # Create the optionmenu widget and passing 
        # the options_list and value_inside to it.
        question_menu1 = tkinter.OptionMenu(root, value_inside1, *options_list1)
        question_menu11 = tkinter.OptionMenu(root, value_inside11, *options_list1)
        question_menu111 = tkinter.OptionMenu(root, value_inside111, *options_list1)
        
        question_menu1.pack()
        question_menu11.pack()
        question_menu111.pack()
        
        value_inside2 = tkinter.StringVar(root)
        # Set the default value of the variable
        value_inside2.set("Select sub-set size of PauliRGB image")
        # Create the optionmenu widget and passing 
        # the options_list and value_inside to it.
        question_menu2 = tkinter.OptionMenu(root, value_inside2, *options_list2)
        question_menu2.pack()
        
        value_inside3 = tkinter.StringVar(root)
        # Set the default value of the variable
        value_inside3.set("Select window size")
        # Create the optionmenu widget and passing 
        # the options_list and value_inside to it.
        question_menu3 = tkinter.OptionMenu(root, value_inside3, *options_list3)
        question_menu3.pack()
        
        value_inside4 = tkinter.StringVar(root)
        # Set the default value of the variable
        value_inside4.set("Select type of pass")
        # Create the optionmenu widget and passing 
        # the options_list and value_inside to it.
        question_menu4 = tkinter.OptionMenu(root, value_inside4, *options_list4.keys())
        question_menu4.pack()
        # Function to print the submitted option-- testing purpose
        def print_answers():
            print("Selected parameter1: {}".format(value_inside1.get()))
            print("Selected parameter2: {}".format(value_inside11.get()))
            print("Selected parameter3: {}".format(value_inside111.get()))
            print("Selected size of pauliRGB image : {}".format(value_inside2.get()))
            print("Selected window size: {}".format(value_inside3.get()))
            print("Selected pass: {}".format(value_inside4.get()))
          
        # Submit button
        # Whenever we click the submit button, our submitted
        # option is printed ---Testing purpose
        submit_button = tkinter.Button(root, text='Submit', command=print_answers)
        submit_button.pack()
        root.mainloop()
        return [value_inside1.get(),value_inside11.get(),value_inside111.get(),value_inside2.get(),value_inside3.get(),options_list4[value_inside4.get()]]

if __name__=='__main__':
   z= ui.maingui()