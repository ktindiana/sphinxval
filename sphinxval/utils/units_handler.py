import os
import sys
from astropy import units as u
import math

__version__ = "0.1" #2022-02-14
__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

def about_units_handler():
    """ units_handler.py handles the units of all the validated
        quantities. For validation, the codes compare only
        quantities that have the same energy channels and flux
        thresholds. Comparing units saved as strings can be problematic
        if the strings aren't written in exactly the same way.
        
        e.g. MeV-1*cm-2*s-1*sr-1 or sr-1*cm-2*s-1*MeV-1 will not
        be recongnized as the same units.
        
        Applying astropy.units to solve this issue.
        
        This code currently doesn't support converting between units,
        e.g. GeV-1*cm-2*s-1*sr-1 to MeV-1*cm-2*s-1*sr-1.
        Will add such functionality if find that it becomes necessary.
        
        .. code-block:: python
        
            subdict1 = dict['sep_forecast_submission']['forecasts']['event_lengths']
            subdict2 = subdict1[1]
            desired_val = subdict2['start_time']
            
    """

#Define the basic units used in validation
diff_flux = u.MeV**-1*u.cm**-2*u.sr**-1*u.s**-1
diff_fluence = u.MeV**-1*u.cm**-2*u.sr**-1
int_flux = u.cm**-2*u.sr**-1*u.s**-1 #pfu
int_fluence = u.cm**-2*u.sr**-1

#conversion for steradians
sr_conv = 4*math.pi

def convert_string_to_units(str_units):
    """ Take units written as a string following the CCMC SEP
        Scoreboard format and convert to astropy units.
        
        Expect, e.g.
        "MeV^-1*s^-1*cm^-2*sr^-1"
        "MeV"
        "cm^-2*sr^-1"
        "pfu"
        
    """
    
    str_units = str_units.replace("*",".")
    str_units = str_units.replace("^","")
    if str_units == "pfu":
        units = int_flux
    else:
        units = u.Unit(str_units)
    
    return units
            
   
def convert_units_to_string(units):
    """ Convert astropy units object to a string.
    """
    return str(units)

   
def convert_to_common_units(str_units):
    """ Take units written as a string following the CCMC SEP
        Scoreboard format and convert to astropy units and
        output as strings.
        
        In this case, basically use astropy.units to shuffle
        the units so that they are always in the same order.
        
        Expect, e.g.
        "MeV^-1*s^-1*cm^-2*sr^-1"
        "MeV"
        "cm^-2*sr^-1"
        "pfu"
        
        INPUTS:
        
        :str_units: (string) units in CCMC SEP Scoreboard format
        
        OUTPUTS:
        
        :conv_units: (string) reorganized units in CCMC SEP Scoreboard
            format
        
    """
    
    units = convert_string_to_units(str_units)
        
    if units == int_flux:
        return "pfu"
    else:
        common_units = units.to_string("cds") #'MeV-1.s-1.sr-1.cm-2'
        common_units.replace(".","*")
    
    return common_units
    
    
def calc_conversion_factor(x_units, y_units):
    """ Find a conversion factor to convert y_units to
        x_units. Return the conversion factor.
        
        y_units --> x_units
        
        INPUTS:
        
        :x_units: (string) units in CCMC SEP Scoreboard format
        :y_units: (string) units in CCMC SEP Scoreboard format
        
        OUTPUT:
        
        :conv: (float) conversion factor
    """
    
    xu = convert_string_to_units(x_units)
    yu = convert_string_to_units(y_units)
    
    #Try to convert y to x with astropy
    #will work for e.g. cm --> m or GeV --> MeV
    try:
        conv = yu.to(xu)
        return conv
    except:
        pass
    
    #Astropy couldn't do the conversion
    #likely caused by a difference in sr for the units
    #we are dealing with
    test = xu/yu
    if test == u.sr: #y-->x is sr
        conv = sr_conv #4pi sr
        return conv
    
    if test == u.Unit(u.sr**-1): #y-->x is 1/sr
        conv = 1./sr_conv #1/4pi 1/sr
        return conv
    
    #No conversion found
    conv = None
    return conv
    
    
    
    

    
        

