#Code for classes
#Forecast Class
#Observation Class
#Matching Class
from . import validation_json_handler as vjson
from . import units_handler as vunits
from . import object_handler as objh
from . import config as cfg
import datetime
import pandas as pd

__version__ = "0.1"
__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" Classes to manage SEP model forecasts and prepared observations.

"""


#####CLASSES############
#------ TRIGGERS ------
class CME:
    def __init__(self, cme_id, start_time, liftoff_time, lat, lon, pa,
                half_width, coordinates, speed, catalog, catalog_id):
        """
        Input:
            :self: (object) CME object
            :cme_id: (string) unique id for CME (useful if multiple CMEs
                used as triggers for a forecast)
            :start_time: (datetime)
            :liftoff_time: (datetime)
            :lat: (float) latitude of CME source eruption
            :lon: (float) longitude of CME source eruption
            :pa: (float) position angle
            :half_width: (float) half width of CME cone
            :speed: (float) CME speed
            :catalog: (string) CME catalog
            :catalog_id: (string) id of CME catalog
        
        Output: a CME object
        
        """
        self.label = 'cme'
        self.id = cme_id
        self.start_time = start_time
        self.liftoff_time = liftoff_time
        self.lat = lat
        self.lon = lon
        self.pa = pa
        self.half_width = half_width
        self.speed = speed
        self.coordinates = coordinates
        self.catalog = catalog
        self.catalog_id = catalog_id
        self.cme_allowed_tags = ['start_time', 'liftoff_time', 'lat',
                'lon', 'pa', 'half_width', 'speed', 'coordinates',
                'catalog', 'catalog_id']
 
        return
        
    def SetValues(self, cme_id, start_time, liftoff_time, lat, lon, pa,
                half_width, coordinates, speed, catalog, catalog_id):
        
        self.id = cme_id
        self.start_time = start_time
        self.liftoff_time = liftoff_time
        self.lat = lat
        self.lon = lon
        self.pa = pa
        self.half_width = half_width
        self.speed = speed
        self.coordinates = coordinates
        self.catalog = catalog
        self.catalog_id = catalog_id
        
        return


class CME_Simulation:
    def __init__(self, cmesim_id, model, sim_completion_time):
        """
        Input:
            :model: (string) model
            :sim_completion_time: (datetime) simulation completion time
            
        Output: a CME_Simulation object
        
        """
        self.label = 'cme_simulation'
        self.id = cmesim_id
        self.model = model
        self.completion_time = sim_completion_time
        
        return

    def SetValues(self, cmesim_id, model, sim_completion_time):
        self.id = cmesim_id
        self.model = model
        self.completion_time = sim_completion_time
        
        return


class Flare:
    def __init__(self, flare_id, last_data_time, start_time, peak_time, \
        end_time, location, lat, lon, intensity, integrated_intensity, \
        noaa_region):
        """
        Input:
            :last_data_time: (datetime)
            :start_time: (datetime)
            :peak_time: (datetime)
            :end_time: (datetime)
            :location: (string) location of flare source eruption N00W00
            :lat: (int) latitude
            :lon: (int) longitude
            :intensity: (float) X-ray intensity of flare at last_data_time
            :integrated_intensity: (float) X-ray intensity summed from start  
                to last
            :noaa_region: (string) identifier of NOAA active region
       
        Ouput: a Flare object
            
        """
        self.label = 'flare'
        self.id = flare_id
        self.last_data_time = last_data_time
        self.start_time = start_time
        self.peak_time = peak_time
        self.end_time = end_time
        self.location = location
        self.lat = lat
        self.lon = lon
        self.intensity = intensity
        self.integrated_intensity = integrated_intensity
        self.noaa_region = noaa_region
        
        return
        
    def SetValues(self, flare_id, last_data_time, start_time, peak_time, end_time, \
        location, lat, lon, intensity, integrated_intensity, noaa_region):
        self.id = flare_id
        self.last_data_time = last_data_time
        self.start_time = start_time
        self.peak_time = peak_time
        self.end_time = end_time
        self.location = location
        self.lat = lat
        self.lon = lon
        self.intensity = intensity
        self.integrated_intensity = integrated_intensity
        self.noaa_region = noaa_region
        
        return
            

class Particle_Intensity:
    def __init__(self, part_id, observatory, instrument, last_data_time,
        ongoing_events):
        """
        Inputs:
            :part_id: (string) unique identifier for this measurement
            :observatory: (string)
            :instrument: (string)
            :last_data_time: (datetime)
            :ongoing_events: (array)
            
        Ouput: a Particle_Intensity object
        
        """
        self.label = 'particle_intensity'
        self.id = part_id
        self.observatory = observatory
        self.instrument = instrument
        self.last_data_time = last_data_time
        self.ongoing_events = ongoing_events
        
        return
        
    def SetValues(part_id, observatory, instrument, last_data_time,
        ongoing_events):
        self.id = part_id
        self.observatory = observatory
        self.instrument = instrument
        self.last_data_time = last_data_time
        self.ongoing_events = ongoing_events
    
        return


##INPUTS
class Magnetic_Connectivity:
    def __init__(self, magcon_id, method, lat, lon, connection_angle,
        solar_wind):
        """
        Input:
            :method: (string)
            :lat: (float)
            :lon: (float)
            :connection_angle: (dict)
            :solar_wind: (dict)
            
        Output: A Magnetic_Connectivity object
        
        """
        self.label = 'magnetic_connectivity'
        self.id = magcon_id
        self.method = method
        self.lat = lat
        self.lon = lon
        self.connection_angle = connection_angle
        self.solar_wind = solar_wind
        
        return
        
    def SetValues(self, magcon_id, method, lat, lon, connection_angle,
        solar_wind):
        self.id = magcon_id
        self.method = method
        self.lat = lat
        self.lon = lon
        self.connection_angle = connection_angle
        self.solar_wind = solar_wind
        
        return



class Magnetogram:
    def __init__(self, magneto_id, observatory, instrument, products):
        """
        Input:
            :magneto_id: (string) unique identifier for magnetogram entry
            :observatory: (string)
            :instrument: (string)
            :products: (array of dict)
        
        
        products has the format e.g.,
        [{"product":"hmi.M_45s_nrt","last_data_time":"2022-03-26T00:59Z"},
         {"product":"hmi.sharp_720s_nrt","last_data_time":"2022-03-26T00:58Z"}]
        
        Output: A Magnetogram object
        
        """
        self.label = 'magnetogram'
        self.id = magneto_id
        self.observatory = observatory
        self.instrument = instrument
        self.products = products
    
        return


class Coronagraph:
    def __init__(self, corona_id, observatory, instrument, products):
        """
        Input:
            :corona_id: (string) unique identifier for coronagraph entry
            :observatory: (string)
            :instrument: (string)
            :products: (array of dict)
        
        Output: A Coronagraph object
        
        """
        self.label = 'coronagraph'
        self.id = corona_id
        self.observatory = observatory
        self.instrument = instrument
        self.products = products
    
        return




#------ FORCAST OR OBSERVATION VALUES -----
##Classes for all the types of values in the Observation and Forecasts
class All_Clear:
    def __init__(self, all_clear, threshold, threshold_units,
                probability_threshold):
        """
        Input:
            :self: (object) All_Clear object
            :all_clear: (boolean) all clear value
            :threshold: (float) threshold applied to get all clear value
            :threshold_units: (astropy units)
            :probability_threshold: (float) threshold applied to derive
                all clear from probabilistic model
        
        Output: an All_Clear object
        
        """
        self.label = 'all_clear'
        self.all_clear_boolean = all_clear
        self.threshold = threshold
        self.threshold_units = threshold_units
        self.probability_threshold = probability_threshold
        self.allowed_tags = ['all_clear_boolean', 'threshold', 'threshold_units', 'probability_threshold']
    
        return

    def SetValues(self, all_clear, threshold, threshold_units,
        probability_threshold):
        self.all_clear = all_clear
        self.threshold = threshold
        self.threshold_units = threshold_units
        self.probability_threshold = probability_threshold
        
        return


class Peak_Intensity:
    def __init__(self, intensity, units, uncertainty, uncertainty_low,
        uncertainty_high, time):
        """
        Input:
            :self: (object) Peak_Instensity object
            :intensity: (float) onset peak intensity value
            :units: (astropy) units
            :uncertainty: (float)
            :uncertainty_low: (float)
            :uncertainty_high: (float)
            :time: (datetime)
        
        Output: a Peak_Intensity object
        
        """
        self.label = 'peak_intensity'
        self.intensity = intensity
        self.units = units
        self.uncertainty = uncertainty
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        self.time = time
        
        return

class Peak_Intensity_Max:
    def __init__(self, intensity, units, uncertainty, uncertainty_low,
        uncertainty_high, time):
        """
        Input:
            :self: (object) Peak_Instensity object
            :intensity: (float) onset peak intensity value
            :units: (astropy) units
            :uncertainty: (float)
            :uncertainty_low: (float)
            :uncertainty_high: (float)
            :time: (datetime)
        
        Output: a Peak_Intensity object
        
        """
        self.label = 'peak_intensity_max'
        self.intensity = intensity
        self.units = units
        self.uncertainty = uncertainty
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        self.time = time
        
        return


class Event_Length:
    def __init__(self, start_time, end_time, threshold, threshold_units):
        """
        Input:
            :self: (object) Event_Length object
            :start_time: (datetime) start of SEP event
            :end_time: (datetime) end of SEP event
            :threshold: (float) threshold to determine start of SEP event
            :threshold_units: (astropy) units
        
        Output: an Event_Length object
        
        """
        self.label = 'event_length'
        self.start_time = start_time
        self.end_time = end_time
        self.threshold = threshold
        self.threshold_units = threshold_units
        
        return
        

class Fluence:
    def __init__(self, id, fluence, units, threshold, threshold_units,
        uncertainty_low, uncertainty_high):
        """
        Input:
            :self: (object) Fluence object
            :id: (string) unique id
            :fluence: (float)
            :units: (astropy) units
            :uncertainty_low: (float)
            :uncertainty_high: (float)
        
        Output: a Fluence object
        
        """
        self.label = 'fluence'
        self.id = id
        self.fluence = fluence
        self.units = units
        self.threshold = threshold
        self.threshold_units = threshold_units
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        
        return
        
class Fluence_Spectrum:
    def __init__(self, start_time, end_time, threshold_start, threshold_end, threshold_units, fluence_units, fluence_spectrum):
        """
        Input:
            :self: (object) Fluence_Spectrum object
            :id: (string) unique id
            :start_time: (datetime) start of SEP event
            :end_time: (datetime) end of SEP event
            :threshold_start: (float) threshold to determine start of SEP event
            :threshold_end: (float) threshold to determine end of
                SEP event; if not present, then assume
                threshold_end = threshold_start
            :threshold_units: (astropy units)
            :fluence_units: (astropy) units
            :fluence_spectrum: (array of dict)
            e.g. [{"energy_min": 5.0, "energy_max": -1, "fluence": 78527636.38502692}, {"energy_min": 10.0, "energy_max": -1, "fluence": 46371821.92788475}, {"energy_min": 30.0, "energy_max": -1, "fluence": 16355421.889077082}, {"energy_min": 50.0, "energy_max": -1, "fluence": 7673363.706302568}, {"energy_min": 60.0, "energy_max": -1, "fluence": 5425386.382761811}, {"energy_min": 100.0, "energy_max": -1, "fluence": 2085984.6018625232}, {"energy_min": 700.0, "energy_max": -1, "fluence": 187.6881309476662}]}]
 
        Output: a Fluence_Spectrum object
        
        """
        self.label = 'fluence_spectrum'
        self.start_time = start_time
        self.end_time = end_time
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.threshold_units = threshold_units
        self.fluence_units = fluence_units
        self.fluence_spectrum = fluence_spectrum
        
        return


class Threshold_Crossing:
    def __init__(self, crossing_time, uncertainty, threshold, threshold_units):
        """
        Input:
            :self: (object) Threshold_Crossing object
            :crossing_time: (datetime) start of SEP event
            :threshold: (float) threshold to determine start of SEP event
            :threshold_units: (astropy units)

        Output: a Threshold_Crossing object
        
        """
        self.label = 'threshold_crossing'
        self.crossing_time = crossing_time
        self.uncertainty = uncertainty
        self.threshold = threshold
        self.threshold_units = threshold_units
        
        return


class Probability:
    def __init__(self, probability_value, uncertainty, threshold,
        threshold_units):
        """
        Input:
            :self: (object) Threshold_Crossing object
            :probability: (float)
            :uncertainty: (float)
            :threshold: (float) threshold to determine start of SEP event
            :threshold_units: (astropy units)

        Output: a Probability object
        
        """
        self.label = 'probability'
        self.probability_value = probability_value
        self.uncertainty = uncertainty
        self.threshold = threshold
        self.threshold_units = threshold_units
        
        return




##-----FORECAST CLASS------
class Forecast():
    def __init__(self, energy_channel):
        
        self.label = 'forecast'
        self.energy_channel = energy_channel #dict
        self.short_name = None
        self.issue_time = None
        self.valid = None #indicates whether prediction window starts
                          #at the same time or after triggers/inputs

        
        #General info
        self.species = None
        self.location = None
        self.prediction_window_start = None
        self.prediction_window_end = None
        
        
        #Triggers
        self.cmes = []
        self.cme_simulations = []
        self.flares = []
        self.particle_intensities = []
        
        #Inputs
        self.magnetic_connectivity = []
        self.magnetograms = []
        self.coronagraphs = []
        
        
        #Forecasts
        self.source = None #source from which forcasts ingested
                        #JSON filename or perhaps database in future
        self.path = None #Path to JSON file
        self.all_clear = All_Clear(None, None, None, None) #All_Clear object
        self.peak_intensity = Peak_Intensity(None, None, None, None, None, None) #Peak_Intensity object
        self.peak_intensity_max = Peak_Intensity_Max(None, None, None, None, None, None) #Peak_Intensity object
        self.event_lengths = []
        self.fluences = []
        self.fluence_spectra = []
        self.threshold_crossings = []
        self.probabilities = []
        self.sep_profile = None

        return

    def check_energy_channel_format(self):
        """ Check energy_channel entries in appropriate formats
            energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
            
            Converts min and max to float and units to astropy units.
            
            This should only be called AFTER all the triggers, inputs,
            and forecasts have been loaded because the energy_channel
            needs to be in the same format as the original CCMC json
            dictionary for the subroutines to work.
            
        """
        if isinstance(self.energy_channel['min'], str):
            self.energy_channel['min'] = float(self.energy_channel['min'])
        if isinstance(self.energy_channel['max'], str):
            self.energy_channel['max'] = float(self.energy_channel['max'])
        if isinstance(self.energy_channel['units'], str):
            self.energy_channel['units'] =\
                vunits.convert_string_to_units(self.energy_channel['units'])

        return


    ## -------TRIGGERS
    def add_triggers_from_dict(self, full_json):
        """ Fills in trigger objects.
            
        """
        is_good, dataD = vjson.check_forecast_json(full_json, self.energy_channel)
        if not is_good: return
        
        if 'triggers' in full_json['sep_forecast_submission']:
            trig_arr = full_json['sep_forecast_submission']['triggers']
        else:
            return
        
        for trig in trig_arr:
            if 'cme' in trig and 'cme_simulation' not in trig:
                start_time, liftoff_time, lat, lon, pa, half_width,\
                speed, coordinates, catalog, catalog_id =\
                    vjson.dict_to_cme(trig['cme'])
                cme = CME("id", start_time, liftoff_time, lat, lon, pa,
                    half_width, coordinates, speed, catalog, catalog_id)
                self.cmes.append(cme)
                continue
            
            if 'cme_simulation' in trig:
                model,sim_completion_time = vjson.dict_to_cme_sim(trig['cme_simulation'])
                cme_sim = CME_Simulation("id", model,sim_completion_time)
                self.cme_simulations.append(cme_sim)
                continue
 
            if 'flare' in trig:
                (last_data_time, start_time, peak_time, end_time,
                 location, lat, lon, intensity, integrated_intensity, noaa_region) = vjson.dict_to_flare(trig['flare'])
                flare = Flare("id", last_data_time, start_time, peak_time,
                            end_time, location, lat, lon, intensity,
                            integrated_intensity, noaa_region)
                self.flares.append(flare)
                continue
 
            if 'particle_intensity' in trig:
                observatory, instrument, last_data_time, \
                ongoing_events = vjson.dict_to_particle_intensity(trig['particle_intensity'])
                pi = Particle_Intensity("id", observatory, instrument,
                    last_data_time, ongoing_events)
                self.particle_intensities.append(pi)
 

    ## -----INPUTS
    def add_inputs_from_dict(self,full_json):
        """ Fills in input objects.
        
        """
        is_good, dataD = vjson.check_forecast_json(full_json, self.energy_channel)
        if not is_good: return
        
        if 'inputs' in full_json['sep_forecast_submission']:
            input_arr = full_json['sep_forecast_submission']['inputs']
        else:
            return

        for input in input_arr:
            if 'magnetic_connectivity' in input:
                method, lat, lon, connection_angle, solar_wind =\
                    vjson.dict_to_mag_connectivity(input['magnetic_connectivity'])
                magcon = Magnetic_Connectivity("id", method, lat, lon,
                    connection_angle, solar_wind)
                self.magnetic_connectivity.append(magcon)
                continue
 
            if 'magnetogram' in input:
                observatory, instrument, products = vjson.dict_to_magnetogram(input['magnetogram'])
                magneto = Magnetogram("id", observatory, instrument, products)
                self.magnetograms.append(magneto)
                continue

            if 'coronagraph' in input:
                observatory, instrument, products = \
                    vjson.dict_to_coronagraph(input['coronagraph'])
                corona = Coronagraph("id", observatory, instrument,
                        products)
                self.coronagraphs.append(corona)




    ## -----FORECASTED QUANTITIES
    def add_forecasts_from_dict(self, full_json):
        """ Extracts appropriate energy channel block and extracts
                forecasts and fills all possible forecasted values
 
        """
        is_good, dataD = vjson.check_forecast_json(full_json, self.energy_channel)
        if not is_good: return
        
        self.short_name = full_json['sep_forecast_submission']['model']['short_name']
        issue_time = full_json['sep_forecast_submission']['issue_time']
        if isinstance(issue_time,str):
            issue_time = vjson.zulu_to_time(issue_time)
        self.issue_time = issue_time
        
        if 'filename' in full_json:
            self.source = full_json['filename']
            
            #pull out path
            fullpath = full_json['filename']
            fullpath = fullpath.strip().split("/")
            if fullpath[0] == "/":
                svpath = "/"
            else:
                svpath = ""
            for x in fullpath:
                if ".json" in x: continue
                svpath = svpath + x + "/"
                
            self.path = svpath
        
        #Supporting information
        if dataD != {}:
            if 'species' in dataD:
                self.species = dataD['species']
            
            if 'location' in dataD:
                self.location = dataD['location']
            
            if 'sep_profile' in dataD:
                self.sep_profile = dataD['sep_profile']
                if 'ASPECS' in self.short_name:
                    pth = self.path
                    pth = pth.split("/")
                    self.sep_profile = "../../../Profile/" + pth[len(pth)-3] + "/" + pth[len(pth)-2] + "/" + dataD['sep_profile']
            
            if 'prediction_window' in dataD:
                self.prediction_window_start = dataD['prediction_window']['start_time']
                self.prediction_window_end = dataD['prediction_window']['end_time']
            
            if isinstance(self.prediction_window_start, str):
                self.prediction_window_start =\
                    vjson.zulu_to_time(self.prediction_window_start)

            if isinstance(self.prediction_window_end, str):
                self.prediction_window_end =\
                    vjson.zulu_to_time(self.prediction_window_end)

        
        #Only add objects of the predicted values are present in the json
        #Load All Clear
        all_clear, threshold, threshold_units, probability_threshold = \
                vjson.dict_to_all_clear(dataD)
        if all_clear != None:
            self.all_clear = All_Clear(all_clear, threshold, threshold_units,
                probability_threshold)

        #Load (Onset) Peak Intensity
        intensity, units, uncertainty, uncertainty_low, uncertainty_high,\
            time = vjson.dict_to_peak_intensity('peak_intensity', dataD)
        if intensity != None:
            self.peak_intensity = Peak_Intensity(intensity, units,
                uncertainty, uncertainty_low, uncertainty_high, time)
            
        #Load Max Intensity
        intensity, units, uncertainty, uncertainty_low, uncertainty_high,\
            time = vjson.dict_to_peak_intensity('peak_intensity_max', dataD)
        if intensity != None:
            self.peak_intensity_max = Peak_Intensity_Max(intensity, units,
                uncertainty, uncertainty_low, uncertainty_high, time)
        
        #Load Event Lengths
        if 'event_lengths' in dataD:
            for event in dataD['event_lengths']:
                start_time, end_time, threshold, threshold_units,=\
                    vjson.dict_to_event_length(event)
                if start_time != None:
                    self.event_lengths.append(Event_Length(start_time,
                        end_time, threshold, threshold_units))
        

        #Load Fluence
        if 'fluences' in dataD:
            for i in range(len(dataD['fluences'])):
                event = {}
                if 'event_lengths' in dataD:
                    event = dataD['event_lengths'][i]

                fl = dataD['fluences'][i]
                fluence, units, threshold, threshold_units, uncertainty_low,\
                    uncertainty_high = vjson.dict_to_fluence(event, fl)
                    
                if 'event_lengths' not in dataD:
                    threshold = self.all_clear.threshold
                    threshold_units = self.all_clear.threshold_units
                
                self.fluences.append(Fluence("id", fluence, units,
                    threshold, threshold_units, uncertainty_low, uncertainty_high))


        #Load Fluence Spectra
        if 'fluence_spectra' in dataD:
            for spectrum in dataD['fluence_spectra']:
                start_time, end_time, threshold_start, threshold_end,\
                threshold_units, fluence_units, fluence_spectrum =\
                    vjson.dict_to_fluence_spectrum(spectrum)
                if fluence_spectrum != None:
                    self.fluence_spectra.append(Fluence_Spectrum(start_time,
                        end_time, threshold_start, threshold_end,
                        threshold_units, fluence_units, fluence_spectrum))


        #Load Threshold Crossings
        if 'threshold_crossings' in dataD:
            for cross in dataD['threshold_crossings']:
                crossing_time, uncertainty, threshold, \
                threshold_units = vjson.dict_to_threshold_crossing(cross)
                if crossing_time != None:
                    self.threshold_crossings.append(Threshold_Crossing(
                    crossing_time, uncertainty, threshold, threshold_units))


        #Load Probabilities
        if 'probabilities' in dataD:
            for prob in dataD['probabilities']:
                probability_value, uncertainty, threshold,\
                threshold_units = vjson.dict_to_probability(prob)
                if probability_value != None:
                    self.probabilities.append(Probability(probability_value,
                        uncertainty, threshold, threshold_units))
                    
        return



    def identify_all_thresholds(self):
        """ Find all the thresholds applied to a given energy channel.
            Thresholds are applied in:
            All clear
            Event lengths
            Fluence spectra
            Threshold crossings
            Probabilities
        
        Inputs:
        
            :obj: (single Forecast or Observation object)
            
        Outputs:
        
            :all_thresholds: (array of dict)
                [{'threshold': 10, 'threshold_units': Unit('pfu')}]
        
        """
        all_thresholds = []
        
        if self.all_clear != None:
            thresh = self.all_clear.threshold
            units = self.all_clear.threshold_units
            if thresh != None and units != None:
                dict = {'threshold':thresh, 'threshold_units': units}
                if dict not in all_thresholds:
                    all_thresholds.append(dict)
        
        if self.event_lengths != []:
            for entry in self.event_lengths:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
        
        if self.fluence_spectra != []:
            for entry in self.fluence_spectra:
                thresh = entry.threshold_start
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
       
        if self.threshold_crossings != []:
            for entry in self.threshold_crossings:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)


        if self.probabilities != []:
            for entry in self.probabilities:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)

        return all_thresholds




    def last_trigger_time(self):
        """ Out of all the triggers, find the last data time
            relevant for matching to observations.
            
            Matching is guided by the idea that that a forecast
            is only valid for time periods after the triggers.
            e.g. if there are multiple CMEs in a simulation,
            the forecast is only relevant for what is observed
            after the eruption of the last CME because observational
            truth has been available to the forecaster/mode up until
            that time.
            
        """
        last_time = None
        last_eruption_time = None #flares and CMEs
        
        #Find the time of the latest CME in the trigger list
        last_cme_time = None
        if self.cmes != []:
            for cme in self.cmes:
                #start time and liftoff time could be essentially
                #the same time and both are indicators of when
                #the CME first erupted. Take the earliest of the
                #two times for matching.
                check_time = None
                start_time = cme.start_time
                liftoff_time = cme.liftoff_time
                
                if start_time == None and liftoff_time == None:
                    continue
                
                if isinstance(start_time,datetime.date):
                    check_time = start_time
                    
                if isinstance(liftoff_time,datetime.date):
                    check_time = liftoff_time
                
                if isinstance(start_time,datetime.date) and\
                    isinstance(liftoff_time,datetime.date):
                    check_time = min(start_time,liftoff_time)
                
                if last_cme_time == None:
                    last_cme_time = check_time
                elif isinstance(check_time,datetime.date):
                    last_cme_time = max(last_cme_time,check_time)
     

        #Find the time of the latest flare in the trigger list
        last_flare_time = None
        if self.flares != []:
            for flare in self.flares:
                #The flare peak time is the most relevant for matching
                #to SEP events as the CME (if any) is often launched
                #around the time of the peak.
                check_time = None
                start_time = flare.start_time
                peak_time = flare.peak_time
                end_time = flare.end_time
                last_data_time = flare.last_data_time
                
                if isinstance(peak_time,datetime.date):
                    check_time = peak_time
                elif isinstance(start_time,datetime.date):
                    check_time = start_time
                elif isinstance(end_time,datetime.date):
                    check_time = end_time
                elif isinstance(last_data_time,datetime.date):
                    check_time = last_data_time
                    
                if last_flare_time == None:
                    last_flare_time = check_time
                elif insinstance(check_time, datetime.date):
                    last_flare_time = max(last_flare_time,check_time)

        #Find the latest particle intensity data used by the model
        last_pi_time = None
        if self.particle_intensities != []:
            for pi in self.particle_intensities:
                check_time = pi.last_data_time
                if isinstance(check_time,datetime.date):
                    if last_pi_time == None:
                        last_pi_time = check_time
                    else:
                        last_pi_time = max(last_pi_time,check_time)


        #Take the latest of all the times
        if isinstance(last_cme_time,datetime.date):
            last_time = last_cme_time
            last_eruption_time = last_cme_time
            
        if isinstance(last_flare_time,datetime.date):
            if last_time == None:
                last_time = last_flare_time
                last_eruption_time = last_flare_time
            else:
                last_time = max(last_time,last_flare_time)
                last_eruption_time = max(last_eruption_time,last_flare_time)
                
        if isinstance(last_pi_time,datetime.date):
            if last_time == None:
                last_time = last_pi_time
            else:
                last_time = max(last_time,last_pi_time)
                
        return last_eruption_time, last_time



    def last_input_time(self):
        """ Out of all the inputs, find the last data time
            relevant for matching to observations.
            
            Matching is guided by the idea that that a forecast
            is only valid for time periods after the last input.

        """
        last_time = None
        
        #Find time of last magnetogram used as input
        last_magneto_time = None
        if self.magnetograms != []:
            for magneto in self.magnetograms:
                if magneto.products == []: continue
                if magneto.products == None: continue
                for prod in magneto.products:
                    last_data_time = prod['last_data_time']
                    if isinstance(last_data_time,datetime.date):
                        if last_magneto_time == None:
                            last_magneto_time = last_data_time
                        else:
                            last_magneto_time = max(last_magneto_time,last_data_time)
                    
        #Find time of last coronagraph used as input
        last_corona_time = None
        if self.coronagraphs != []:
            for corona in self.coronagraphs:
                if corona.products == []: continue
                if corona.products == None: continue
                for prod in corona.products:
                    last_data_time = prod['last_data_time']
                    if isinstance(last_data_time,datetime.date):
                        if last_corona_time == None:
                            last_corona_time = last_data_time
                        else:
                            last_corona_time = max(last_corona_time,last_data_time)

        if isinstance(last_magneto_time,datetime.date):
            last_time = last_magneto_time
            
        if isinstance(last_corona_time,datetime.date):
            if last_time == None:
                last_time = last_corona_time
            else:
                last_time = max(last_time,last_corona_time)
        
        return last_time


    def valid_forecast(self, last_trigger_time, last_input_time):
        """ Check that the triggers and inputs are at the same time of
            or before the start of the prediction window. The prediction
            window cannot start before the info required to make
            the forecast.
            
        Input:
        
            :obj: (Forecast object)
            
        Output:
        
            Updated self.valid field
        
        """

        if self.issue_time == None:
            return
            
        if last_trigger_time == None and last_input_time == None:
            return
        
        self.valid = True
        if last_trigger_time != None:
            if self.issue_time < last_trigger_time:
                self.valid = False

        if last_input_time != None:
            if self.issue_time < last_input_time:
                self.valid = False

        return


    def print_forecast_values(self):
        print()
        print("--- " + self.source + " -----")
        print(self.energy_channel)
        print(self.short_name)
        print(self.issue_time)
        print(self.location)
        print(self.species)
        print(self.prediction_window_start)
        print(self.prediction_window_end)
        print(self.sep_profile)
        
        if self.cmes != []:
            for cme in self.cmes:
                print(vars(cme))
        if self.cme_simulations != []:
            for sim in self.cme_simulations:
                print(vars(sim))
        if self.flares != []:
            for flare in self.flares:
                print(vars(flare))
        if self.particle_intensities != []:
            for pi in self.particle_intensities:
                print(vars(pi))
     
        if self.magnetic_connectivity != []:
            for mag in self.magnetic_connectivity:
                print(vars(mag))
        if self.magnetograms != []:
            for magneto in self.magnetograms:
                print(vars(magneto))
        if self.coronagraphs != []:
            for corona in self.coronagraphs:
                print(vars(corona))
     
        print(vars(self.all_clear))
        print(vars(self.peak_intensity))
        print(vars(self.peak_intensity_max))
        if self.event_lengths != []:
            for ev in self.event_lengths:
                print(vars(ev))
        if self.fluences != []:
            for fl in self.fluences:
                print(vars(fl))
        if self.fluence_spectra != []:
            for fls in self.fluence_spectra:
                print(vars(fls))
        if self.threshold_crossings != []:
            for tc in self.threshold_crossings:
                print(vars(tc))
        if self.probabilities != []:
            for prob in self.probabilities:
                print(vars(prob))







##-----OBSERVATION CLASS------
class Observation():
    def __init__(self, energy_channel):
        
        self.label = 'observation'
        self.energy_channel = energy_channel #dict
        self.short_name = None
        self.issue_time = None

        
        #General info
        self.species = None
        self.location = None
        self.observation_window_start = None
        self.observation_window_end = None
        
        
        #Forecasts
        self.all_clear = All_Clear(None, None, None, None) #All_Clear object
        self.peak_intensity = Peak_Intensity(None, None, None, None, None, None) #Peak_Intensity object
        self.peak_intensity_max = Peak_Intensity_Max(None, None, None, None, None, None)#Peak_Intensity object
        self.event_lengths = []
        self.fluences = []
        self.fluence_spectra = []
        self.threshold_crossings = []
        self.sep_profile = None

        return


    def check_energy_channel_format(self):
        """ Check energy_channel entries in appropriate formats
            energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
            
            Converts min and max to float and units to astropy units.
            
            This should only be called AFTER all the triggers, inputs,
            and forecasts have been loaded because the energy_channel
            needs to be in the same format as the original observation json
            dictionary for the subroutines to work.
            
        """
        if isinstance(self.energy_channel['min'], str):
            self.energy_channel['min'] = float(self.energy_channel['min'])
        if isinstance(self.energy_channel['max'], str):
            self.energy_channel['max'] = float(self.energy_channel['max'])
        if isinstance(self.energy_channel['units'], str):
            self.energy_channel['units'] =\
                vunits.convert_string_to_units(self.energy_channel['units'])

        return



    ## -----Observed QUANTITIES
    def add_observations_from_dict(self, full_json):
        """ Extracts appropriate energy channel block and extracts
                forecasts and fills all possible observed values
 
        """
        is_good, dataD = vjson.check_observation_json(full_json, self.energy_channel)
        if not is_good: return
        
        self.short_name = full_json['sep_observation_submission']['observatory']['short_name']
        issue_time = full_json['sep_observation_submission']['issue_time']
        if isinstance(issue_time,str):
            issue_time = vjson.zulu_to_time(issue_time)
        self.issue_time = issue_time
        
        if 'filename' in full_json:
            self.source = full_json['filename']
            

        #Supporting information
        if dataD != {}:
            if 'species' in dataD:
                self.species = dataD['species']
            
            if 'location' in dataD:
                self.location = dataD['location']
            
            if 'sep_profile' in dataD:
                self.sep_profile = dataD['sep_profile']
            
            if 'observation_window' in dataD:
                self.observation_window_start = dataD['observation_window']['start_time']
                self.observation_window_end = dataD['observation_window']['end_time']
            
            if isinstance(self.observation_window_start, str):
                self.observation_window_start =\
                    vjson.zulu_to_time(self.observation_window_start)

            if isinstance(self.observation_window_end, str):
                self.observation_window_end =\
                    vjson.zulu_to_time(self.observation_window_end)

        
        
        #Load All Clear
        all_clear, threshold, threshold_units, probability_threshold = \
                vjson.dict_to_all_clear(dataD)
        self.all_clear = All_Clear(all_clear, threshold, threshold_units,
                probability_threshold)

        #Load (Onset) Peak Intensity
        intensity, units, uncertainty, uncertainty_low, uncertainty_high,\
            time = vjson.dict_to_peak_intensity('peak_intensity', dataD)
        self.peak_intensity = Peak_Intensity(intensity, units, uncertainty,
            uncertainty_low, uncertainty_high, time)
            
        #Load Max Intensity
        intensity, units, uncertainty, uncertainty_low, uncertainty_high,\
            time = vjson.dict_to_peak_intensity('peak_intensity_max', dataD)
        self.peak_intensity_max = Peak_Intensity_Max(intensity, units,
            uncertainty, uncertainty_low, uncertainty_high, time)
        
        #Load Event Lengths
        if 'event_lengths' in dataD:
            for event in dataD['event_lengths']:
                start_time, end_time, threshold, threshold_units,=\
                    vjson.dict_to_event_length(event)
                self.event_lengths.append(Event_Length(start_time, end_time,
                    threshold, threshold_units))

        #Load Fluence
        if 'fluences' in dataD:
            for i in range(len(dataD['fluences'])):
                event = {}
                if 'event_lengths' in dataD:
                    event = dataD['event_lengths'][i]
                fl = dataD['fluences'][i]
                fluence, units, threshold, threshold_units, uncertainty_low,\
                    uncertainty_high = vjson.dict_to_fluence(event, fl)
                    
                if 'event_lengths' not in dataD:
                    threshold = self.all_clear.threshold
                    threshold_units = self.all_clear.threshold_units
                
                self.fluences.append(Fluence("id", fluence, units,
                    threshold, threshold_units, uncertainty_low, uncertainty_high))


        #Load Fluence Spectra
        if 'fluence_spectra' in dataD:
            for spectrum in dataD['fluence_spectra']:
                start_time, end_time, threshold_start, threshold_end,\
                threshold_units, fluence_units, fluence_spectrum =\
                    vjson.dict_to_fluence_spectrum(spectrum)
                self.fluence_spectra.append(Fluence_Spectrum(start_time,
                    end_time, threshold_start, threshold_end,
                    threshold_units, fluence_units, fluence_spectrum))


        #Load Threshold Crossings
        if 'threshold_crossings' in dataD:
            for cross in dataD['threshold_crossings']:
                crossing_time, uncertainty, threshold, \
                threshold_units = vjson.dict_to_threshold_crossing(cross)
                self.threshold_crossings.append(Threshold_Crossing(
                crossing_time, uncertainty, threshold, threshold_units))
                    
        return



    def identify_all_thresholds(self):
        """ Find all the thresholds applied to a given energy channel.
            Thresholds are applied in:
            All clear
            Event lengths
            Fluence spectra
            Threshold crossings
            Probabilities
        
        Inputs:
        
            :obj: (single Forecast or Observation object)
            
        Outputs:
        
            :all_thresholds: (array of dict)
                [{'threshold': 10, 'threshold_units': Unit('pfu')}]
        
        """
        all_thresholds = []
        
        if self.all_clear != None:
            thresh = self.all_clear.threshold
            units = self.all_clear.threshold_units
            if thresh != None and units != None:
                dict = {'threshold':thresh, 'threshold_units': units}
                if dict not in all_thresholds:
                    all_thresholds.append(dict)
        
        if self.event_lengths != []:
            for entry in self.event_lengths:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
        
        if self.fluence_spectra != []:
            for entry in self.fluence_spectra:
                thresh = entry.threshold_start
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
       
        if self.threshold_crossings != []:
            for entry in self.threshold_crossings:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)

        return all_thresholds



    def print_observed_values(self):
        print()
        print("--- " + self.source + " -----")
        print(self.energy_channel)
        print(self.short_name)
        print(self.issue_time)
        print(self.location)
        print(self.species)
        print(self.observation_window_start)
        print(self.observation_window_end)
        print(self.sep_profile)
        print(vars(self.all_clear))
        print(vars(self.peak_intensity))
        print(vars(self.peak_intensity_max))
        if self.event_lengths != []:
            for ev in selfevent_lengths:
                print(vars(ev))
        if self.fluences != []:
            for fl in self.fluences:
                print(vars(fl))
        if self.fluence_spectra != []:
            for fls in self.fluence_spectra:
                print(vars(fls))
        if self.threshold_crossings != []:
            for tc in self.threshold_crossings:
                print(vars(tc))
        if self.probabilities != []:
            for prob in self.probabilities:
                print(vars(prob))





class SPHINX:
    def __init__(self, energy_channel):
        """ A SPHINX object contains the forecasted values and
            matching observed values for validation.
            
            All information used for matching is saved so that the
            logic is completely traceable and readily understood to
            ensure that the correct observed values were matched to the
            forecasted values.
        """
        
        self.label = 'sphinx'
        self.energy_channel = energy_channel #dict
        self.prediction = None #Forecast object

        #MATCHING INFORMATION
        #If user specified in config file to allow the observations
        #and predictions to have two different energy channels and thresholds,
        #self.mismatch will be changed to True to indicate this choice in the
        #matching.
        self.mismatch = False
        #Observations with observations windows that overlap with
        #the prediction windows - first rough cut at matching
        self.prediction_observation_windows_overlap = [] #array of Observation objs
        self.overlapping_indices = [] #indices of observations as they were read in
        self.observed_sep_profiles = [] #Always fill for overlapping observations
        
        self.thresholds = [] #all of the thresholds in the observations
        self.threshold_crossed_in_pred_win = {} #filenames of the
            #observations that satisfy the criteria (self.source)
        self.last_eruption_time = None
        self.last_trigger_time = None
        self.last_input_time = None
        
        #Indicate whether a forecast was originally matched to an SEP event
        #and then unmatched in match.py/revise_eruption_matches()
        #Will set to true if unmatched
        self.unmatched = False

        #Criteria related to observed peak intensity fields
        #Dictionaries organized by threshold and number of overlapping
        #observations {'threshold1': [obs1, obs2], 'threshold2':[obs1, obs2]}
        self.peak_intensity_time_in_prediction_window = []
        self.triggers_before_peak_intensity = []
        self.time_difference_triggers_peak_intensity = [] #hours
        self.inputs_before_peak_intensity = []
        self.time_difference_inputs_peak_intensity = [] #hours

        self.peak_intensity_max_time_in_prediction_window = []
        self.triggers_before_peak_intensity_max = []
        self.time_difference_triggers_peak_intensity_max = [] #hours
        self.inputs_before_peak_intensity_max = []
        self.time_difference_inputs_peak_intensity_max = [] #hours

        #Criteria related to observed threshold crossing times
        self.eruptions_before_threshold_crossing = {}
        self.time_difference_eruptions_threshold_crossing = {}
        self.triggers_before_threshold_crossing = {}
        self.time_difference_triggers_threshold_crossing = {} #hours
        self.inputs_before_threshold_crossing = {}
        self.time_difference_inputs_threshold_crossing = {} #hours

        #Criteria related to SEP end times
        self.triggers_before_sep_end = {}
        self.time_difference_triggers_sep_end = {} #hours
        self.inputs_before_sep_end = {}
        self.time_difference_inputs_sep_end = {} #hours

        self.prediction_window_sep_overlap = {}
        self.observed_ongoing_events = {} #multiple thresholds
 
        #OBSERVED VALUES THAT HAVE BEEN MATCHED TO PREDICTIONS
        #All matched observed values are saved regardless of whether a
        #prediction was made for that value or not.
        #Each observed value is selected using an individual set of criteria
        #for that specific quantity.
        #These criteria are specified in match.py/match_all_forecasts()
        self.observed_match_peak_intensity_source = None
        self.peak_intensity_match_status = ""
        self.observed_peak_intensity = Peak_Intensity(None, None, None, None, None, None) #Peak Intensity Obj
        self.observed_match_peak_intensity_max_source = None
        self.peak_intensity_max_match_status = ""
        self.observed_peak_intensity_max = Peak_Intensity(None, None, None, None, None, None) #Peak Intensity Max Obj
        #Only one All Clear status allowed per energy channel
        self.observed_match_all_clear_source = None
        self.all_clear_match_status = ""
        self.observed_all_clear = All_Clear(None, None, None, None)  #All Clear Object
        #Uses thresholds from self.thresholds as keys
        self.observed_match_sep_source = {}
        self.sep_match_status = {}
        self.observed_threshold_crossing = {} #Threshold Crossing objects
        self.observed_event_length = {} #Event Length objects
        self.observed_start_time = {} #datetime
        self.observed_fluence = {} #Fluence objects
        self.observed_fluence_spectrum = {} #Fluence spectrum objects

        self.end_time_match_status = {}
        self.observed_end_time = {} #datetime

        self.time_profile_match_status = {}
        self.observed_time_profile = {} #string, filename

        #Probability matching status the same as self.sep_match_status
        self.observed_probability_source = {}
        self.observed_probability = {} #Probability object
        
        return


    def add_threshold(self, threshold):
        """ Updated the dictionary values to contain an entry for
            a specific threshold.
            
        """
        key = objh.threshold_to_key(threshold)
        
        self.threshold_crossed_in_pred_win.update({key:[]})
        
        #Criteria related to observed threshold crossing times
        self.eruptions_before_threshold_crossing.update({key:[]})
        self.time_difference_eruptions_threshold_crossing.update({key:[]})
        self.triggers_before_threshold_crossing.update({key:[]})
        self.time_difference_triggers_threshold_crossing.update({key:[]})
        self.inputs_before_threshold_crossing.update({key:[]})
        self.time_difference_inputs_threshold_crossing.update({key:[]})

        #Criteria related to SEP end times
        self.triggers_before_sep_end.update({key:[]})
        self.time_difference_triggers_sep_end.update({key:[]})
        self.inputs_before_sep_end.update({key:[]})
        self.time_difference_inputs_sep_end.update({key:[]})

        self.prediction_window_sep_overlap.update({key:[]})
        self.observed_ongoing_events.update({key:[]})
        
        #Observed values
        self.observed_match_sep_source.update({key:None})
        self.sep_match_status.update({key:""})
        self.observed_threshold_crossing.update({key:Threshold_Crossing(None, None, None, None)})
        self.observed_event_length.update({key: Event_Length(None, None, None, None)})
        self.observed_start_time.update({key:None})
        self.observed_fluence.update({key:Fluence("id",None, None, None, None, None, None)})
        self.observed_fluence_spectrum.update({key:Fluence_Spectrum(None, None, None, None, None, None, None)})

        self.end_time_match_status.update({key:""})
        self.observed_end_time.update({key:None})
        
        self.time_profile_match_status.update({key:""})
        self.observed_time_profile.update({key:None})
        
        self.observed_probability_source.update({key: None})
        self.observed_probability.update({key:Probability(None, None, None, None)})
        
        return



    def match_criteria():
        """ Print matching criteria to match up observations to
            each predicted quantity.

        """
        print("\n")
        print("====== Matching Criteria =======")
        print("\n")
        print("------ Criteria to match with an observed SEP event ----------")
        print("- Prediction window overlaps with observation")
        print("- Last eruption within 48 hrs - 15 mins before threshold "
            "crossing")
        print("- The prediction window overlaps with an SEP event in any " "threshold - only a comparison when there is an SEP event")
        print("- The last trigger/input time if before the observed peak "
            "intensity")

        return



    def match_report(self):
        """ Generate a report describing all of the steps involved in the
            matching and information about the prediction and observations.

        """

        print("\n")
        print("=================== Matching Report =========================")
        print("-------------------------------------------------------------")
        print("Model and Prediction Information")
        print("-------------------------------------------------------------")
        print("  Model: " + self.prediction.short_name)
        print("  Original prediction source: " + self.prediction.source)
        print("  Prediction Issue time: " + str(self.prediction.issue_time))
        print("  Energy Channel: " + str(self.energy_channel))
        print("  User allowed mismatch: " + str(self.mismatch))
        
        print("-------------------------------------------------------------")
        print("All Observations overlapping with the Prediction Window")
        print("-------------------------------------------------------------")
        print("  Prediction Window: "
            + str(self.prediction.prediction_window_start) + " to "
            + str(self.prediction.prediction_window_end))
        print("  Observations that overlapped with Prediction Window:")
        if self.prediction_observation_windows_overlap == []:
            print("  No matching observations were found.")
        else:
            print("  Observation Sources: ")
            for obs in self.prediction_observation_windows_overlap:
                print("  " + obs.source)
            print("  Observed Time Profiles: ")
            for prof in self.observed_sep_profiles:
                print("  " + prof)

        print("-------------------------------------------------------------")
        print("Prediction Eruption/Trigger/Input Timing")
        print("None = no trigger or input used by model")
        print("-------------------------------------------------------------")
        print("  Prediction last eruption time: "
                + str(self.last_eruption_time))
        print("  Prediction last trigger time: " + str(self.last_trigger_time))
        print("  Prediction last input time: " + str(self.last_input_time))

        print("-------------------------------------------------------------")
        print("Thresholds Applied to the Energy Channel")
        print("All values below related to thresholds are in the order of\n"
            "the matched observations first and thresholds second.")
        print("-------------------------------------------------------------")
        if self.thresholds == []:
            print("  No thresholds were present in both predictions and "
                "observations.")
        else:
            for thresh in self.thresholds:
                print("  " + str(thresh))
                
                

        print("-------------------------------------------------------------")
        print("Were thresholds crossed inside of the Prediction Window?")
        print("Entries are in order of observations and thresholds.")
        print("-------------------------------------------------------------")
        if self.threshold_crossed_in_pred_win == []:
            print("  No thresholds were present in both predictions and "
                "observations.")
        else:
            for thresh in self.thresholds:
                thresh_key = objh.threshold_to_key(thresh)
                print(" Threshold: " + str(thresh))
                print(" Match Status: " + self.sep_match_status[thresh_key])
                print("  " + str(self.threshold_crossed_in_pred_win[thresh_key]))
                print("  Observed threshold crossing times (NaT = no threshold crossing present \n"
                    "  that satisfied matching criteria): ")
                print("  " + str(self.observed_threshold_crossing[thresh_key].crossing_time))
            

        print("-------------------------------------------------------------")
        print("Was the Eruption (flare/CME) before Threshold Crossing?")
        print("None = no flare/CME information used by model")
        print("False = flare/CME after threshold crossing")
        print("True = flare/CME before threshold crossing")
        print("-------------------------------------------------------------")
        if self.eruptions_before_threshold_crossing == []:
            print("  No eruption information used by model.")
        else:
            for thresh in self.thresholds:
                thresh_key = objh.threshold_to_key(thresh)
                print(" Threshold: " + str(thresh))
                print("  "
                + str(self.eruptions_before_threshold_crossing[thresh_key]))
                print("  Time difference (hrs) (eruption time - threshold "
                    "crossing time):")
                print("  " +
                str(self.time_difference_eruptions_threshold_crossing[thresh_key]))

        print("-------------------------------------------------------------")
        print("Were all Triggers before Threshold Crossing? ")
        print("None = no triggers used by model or no thresholds crossed")
        print("False = triggers after threshold crossing")
        print("True = triggers before threshold crossing")
        print("-------------------------------------------------------------")
        if self.triggers_before_threshold_crossing == []:
            print("  No triggers used by model.")
        else:
            for thresh in self.thresholds:
                thresh_key = objh.threshold_to_key(thresh)
                print(" Threshold: " + str(thresh))
                print("  " + str(self.triggers_before_threshold_crossing[thresh_key]))

        print("-------------------------------------------------------------")
        print("Were all Inputs before Threshold Crossing?")
        print("None = no inputs used by model or no thresholds crossed")
        print("False = inputs after threshold crossing")
        print("True = inputs before threshold crossing")
        print("-------------------------------------------------------------")
        if self.inputs_before_threshold_crossing == []:
            print("  No inputs used by model.")
        else:
            for thresh in self.thresholds:
                thresh_key = objh.threshold_to_key(thresh)
                print(" Threshold: " + str(thresh))

                print("  " + str(self.inputs_before_threshold_crossing[thresh_key]))
        
        print("-------------------------------------------------------------")
        print("Is the last Trigger before the Onset Peak (peak_intensity)?")
        print("None = no triggers used by model or no SEP event.")
        print("-------------------------------------------------------------")
        if self.triggers_before_peak_intensity == []:
            print("  No triggers used by model.")
        else:
            print("  " + str(self.triggers_before_peak_intensity))
            print("  Time difference (hrs) (last trigger - onset peak time):")
            print("  " + str(self.time_difference_triggers_peak_intensity))

        print("-------------------------------------------------------------")
        print("Is the last Input before the Onset Peak (peak_intensity)?")
        print("None = no inputs used by model or no SEP event.")
        print("-------------------------------------------------------------")
        if self.inputs_before_peak_intensity == []:
            print("  No inputs used by model.")
        else:
            print("  " + str(self.inputs_before_peak_intensity))
            print("  Time difference (hrs) (last input - onset peak time):")
            print("  " + str(self.time_difference_inputs_peak_intensity))

        print("-------------------------------------------------------------")
        print("Is the last Trigger before the Max Flux (peak_intensity_max)?")
        print("**If no SEP event, Max Flux is the maximum value in the observation.")
        print("None = no triggers used by model ")
        print("-------------------------------------------------------------")
        if self.triggers_before_peak_intensity_max == []:
            print("  No triggers used by model.")
        else:
            print("  " + str(self.triggers_before_peak_intensity_max))
            print("  Time difference (hrs) (last trigger - max flux time):")
            print("  " + str(self.time_difference_triggers_peak_intensity_max))

        print("-------------------------------------------------------------")
        print("Is the last Input before the Max Flux (peak_intensity_max)?")
        print("**If no SEP event, Max Flux is the maximum value in the observation.")
        print("None = no inputs used by model ")
        print("-------------------------------------------------------------")
        if self.inputs_before_peak_intensity_max == []:
            print("  No inputs used by model.")
        else:
            print("  " + str(self.inputs_before_peak_intensity_max))
            print("  Time difference (hrs) (last input - max flux time):")
            print("  " + str(self.time_difference_inputs_peak_intensity_max))

        print("-------------------------------------------------------------")
        print("Is the last Trigger before the SEP End Time?")
        print("None = no triggers used by model or no SEP event.")
        print("-------------------------------------------------------------")
        if self.triggers_before_sep_end == []:
            print("  No triggers used by model.")
        else:
            for thresh in self.thresholds:
                thresh_key = objh.threshold_to_key(thresh)
                print(" Threshold: " + str(thresh))

                print("  " + str(self.triggers_before_sep_end[thresh_key]))
                print("  Time difference (hrs) (last trigger - SEP end time):")
                print("  " +
                str(self.time_difference_triggers_sep_end[thresh_key]))

        print("-------------------------------------------------------------")
        print("Is the last Input before the SEP End Time?")
        print("None = no inputs used by model or no SEP event.")
        print("-------------------------------------------------------------")
        if self.inputs_before_sep_end == []:
            print("  No inputs used by model.")
        else:
            for thresh in self.thresholds:
                thresh_key = objh.threshold_to_key(thresh)
                print(" Threshold: " + str(thresh))

                print("  " + str(self.inputs_before_sep_end[thresh_key]))
                print("  Time difference (hrs) (last input - SEP end time):")
                print("  " +
                str(self.time_difference_inputs_sep_end[thresh_key]))

        print("-------------------------------------------------------------")
        print("Does the Prediction Window overlap at all with an observed SEP event?")
        print("-------------------------------------------------------------")
        if self.prediction_window_sep_overlap == []:
            print("  No SEP events.")
        else:
            for thresh in self.thresholds:
                thresh_key = objh.threshold_to_key(thresh)
                print(" Threshold: " + str(thresh))

                print("  " + str(self.prediction_window_sep_overlap[thresh_key]))

        print("-------------------------------------------------------------")
        print("Is there an ongoing observed SEP event at start of prediction window?")
        print("None = no SEP event present in the observation file.")
        print("-------------------------------------------------------------")
        if self.observed_ongoing_events == []:
            print("  No SEP events.")
        else:
            for thresh in self.thresholds:
                thresh_key = objh.threshold_to_key(thresh)
                print(" Threshold: " + str(thresh))

                print("  " + str(self.observed_ongoing_events[thresh_key]))

        print("================= MATCHED OBSERVED VALUES ===================")

        print("-------------------------------------------------------------")
        print("Observed All Clear (True = No SEP, False = SEP)")
        print("If True, will list last matched observation.")
        print("None = ongoing SEP event at start of the prediction window")
        print("-------------------------------------------------------------")
        print("  Matched observation: " + str(self.observed_match_all_clear_source))
        print("  Match Status: " + self.all_clear_match_status)
        print("  Observed All Clear Status: "
            + str(self.observed_all_clear.all_clear_boolean))
        print("  All Clear Threshold: " + str(self.observed_all_clear.threshold))
        print("  All Clear Threshold Units: "
            + str(self.observed_all_clear.threshold_units))

        print("-------------------------------------------------------------")
        print("Observed Onset Peak (peak_intensity)")
        print("None = no SEP event or no match with an observation")
        print("-------------------------------------------------------------")
        print("  Matched observation: " +
            str(self.observed_match_peak_intensity_source))
        print("  Match Status: " + self.peak_intensity_match_status)
        print("  Intensity: " + str(self.observed_peak_intensity.intensity))
        print("  Units: " + str(self.observed_peak_intensity.units))
        print("  Time: " + str(self.observed_peak_intensity.time))

        print("-------------------------------------------------------------")
        print("Observed Max Flux (peak_intensity_max)")
        print("None = no SEP event or no match with an observation")
        print("-------------------------------------------------------------")
        print("  Matched observation: " +
            str(self.observed_match_peak_intensity_max_source))
        print("  Match Status: " + self.peak_intensity_max_match_status)
        print("  Intensity: " + str(self.observed_peak_intensity_max.intensity))
        print("  Units: " + str(self.observed_peak_intensity_max.units))
        print("  Time: " + str(self.observed_peak_intensity_max.time))

        print("-------------------------------------------------------------")
        print("Observed SEP Event Probability: ")
        print("None = no match with an observation, ongoing event, or "
                "threshold crossed in prediction window but the last "
                "trigger or input is after the threshold crossing.")
        print("0.0 = no SEP event (threshold crossing) in prediction window")
        print("1.0 = SEP event (threshold crossing) in prediction window")
        print("-------------------------------------------------------------")
        for thresh in self.thresholds:
            thresh_key = objh.threshold_to_key(thresh)
            print(" Threshold: " + str(thresh))
            print("  Match Status: " + self.sep_match_status[thresh_key])
            print("  Matched observation: " + str(self.observed_probability_source[thresh_key]))
            print("  Probability: "
            + str(self.observed_probability[thresh_key].probability_value))

        print("-------------------------------------------------------------")
        print("Observed SEP Event Characteristics: ")
        print("None = no SEP event or no match with an observation")
        print("NaT = no SEP event or no match with an observation")
        print("-------------------------------------------------------------")
        for thresh in self.thresholds:
            thresh_key = objh.threshold_to_key(thresh)
            print(" Threshold: " + str(thresh))
            print("  Match Status: " + self.sep_match_status[thresh_key])
            print("  Matched observation: " + str(self.observed_match_sep_source[thresh_key]))
            print("  Threshold crossing time: "
                + str(self.observed_threshold_crossing[thresh_key].crossing_time))
            print("  Start time: " + str(self.observed_start_time[thresh_key]))
            print("  Channel fluence: " + str(self.observed_fluence[thresh_key].fluence))
            print("  Fluence Spectrum: " + str(self.observed_fluence_spectrum[thresh_key].fluence_spectrum))

        print("-------------------------------------------------------------")
        print("Observed SEP Event End Times: ")
        print("NaT = no SEP event or no match with an observation")
        print("-------------------------------------------------------------")
        for thresh in self.thresholds:
            thresh_key = objh.threshold_to_key(thresh)
            print(" Threshold: " + str(thresh))
            print("  Match Status: " + self.end_time_match_status[thresh_key])
            print("  Matched observation: " + str(self.observed_match_sep_source[thresh_key]))
            print("  End time: " + str(self.observed_end_time[thresh_key]))
            
        print("-------------------------------------------------------------")
        print("Observed SEP Time Profile: ")
        print("None = no SEP event or no match with an observation")
        print("-------------------------------------------------------------")
        for thresh in self.thresholds:
            thresh_key = objh.threshold_to_key(thresh)
            print(" Threshold: " + str(thresh))
            print("  Match Status: " + self.time_profile_match_status[thresh_key])
            print("  Matched observation: " + str(self.observed_match_sep_source[thresh_key]))
            print("  Time Profile: " + str(self.observed_time_profile[thresh_key]))
                
        print("================== END REPORT ===============================")

        return




    def unmatch(self, threshold):
        """ For forecasts that use eruptions, unamtch from a specific
            SEP event if it has been determined that it is not the best
            matching forecast.
            
            Sets all observed values to None or pd.NaT or whatever the
            outcomes are whenever no SEP event was found.
            
            Switches All Clear to True.
            
        Input:
        
            :self: this SPHINX object
            :threshold: (dict) threshold being unmatched
            
        Output:
        
            None, sphinx object updated
            
        """
        self.unmatch = True
        
        thresh_key = objh.threshold_to_key(threshold)
        
        #These criteria are specified in match.py/match_all_forecasts()
        self.peak_intensity_match_status = "Unmatched"
        self.observed_match_peak_intensity_source = None
        self.observed_peak_intensity = Peak_Intensity(None, None, None, None, None, None)
        self.peak_intensity_max_match_status = "Unmatched"
        self.observed_match_peak_intensity_max_source = None
        self.observed_peak_intensity_max = Peak_Intensity_Max(None, None, None, None, None, None)
        
        #Only one All Clear status allowed per energy channel
        self.all_clear_match_status = "Unmatched"
        self.observed_all_clear.all_clear_boolean = True
        
        #Uses thresholds from self.thresholds as keys
        self.sep_match_status[thresh_key] = "Unmatched"
        self.observed_probability[thresh_key].probability_value = 0.0
        self.observed_threshold_crossing[thresh_key] = Threshold_Crossing(None, None, None, None)
        self.observed_event_length[thresh_key] = Event_Length(None, None, None, None)
        self.observed_start_time[thresh_key] = None
        self.observed_end_time[thresh_key] = None
        self.observed_fluence[thresh_key] = Fluence("id",None, None, None, None, None, None)
        self.observed_fluence_spectrum[thresh_key] = Fluence_Spectrum(None, None, None, None, None, None, None)
        
        return


    def allowed_thresh_mismatch(self, pred_threshold, obs_threshold):
        """ Check if thresholds allowed via config file.
        """
        #If allow mismatching energy channels and thresholds
        if cfg.do_mismatch:
            if cfg.mm_model in self.prediction.short_name:
                if pred_threshold == cfg.mm_pred_threshold and \
                    obs_threshold == cfg.mm_obs_threshold:
                    return True
                    
        return False
    

    def return_predicted_all_clear(self):
        """ Pull out the predicted value.
            Performs units checking with observed values.
            
            None is returned if threshold isn't found or model
            doesn't make prediction.
            
        """
        match_status = ""
        
        #Check if forecast for all clear
        predicted = self.prediction.all_clear.all_clear_boolean
        if predicted == None:
            return predicted, match_status

        pred_threshold = {'threshold': self.prediction.all_clear.threshold,
            'threshold_units': self.prediction.all_clear.threshold_units}
        obs_threshold = {'threshold': self.observed_all_clear.threshold,
            'threshold_units': self.observed_all_clear.threshold_units}
        match_status = self.all_clear_match_status
        
        #If allow mismatching energy channels and thresholds
        if self.mismatch:
            if self.allowed_thresh_mismatch(pred_threshold, obs_threshold):
                return predicted, match_status
        
        #Thresholds must match
        if pred_threshold != obs_threshold:
            predicted = None
            match_status = "No Matching Threshold"
            return predicted, match_status


        return predicted, match_status


    def mismatch_thresh_key(self, pred_thresh):
        """ Check if predicted threshold should be matched up with a
            different observed threshold. Make a new thresh key.
            
        """
        tk = objh.threshold_to_key(pred_thresh)
        if pred_thresh not in self.thresholds:
            if cfg.do_mismatch and cfg.mm_model in self.prediction.short_name:
                if pred_thresh == cfg.mm_pred_threshold:
                    obs_thresh = cfg.mm_obs_threshold
                    if cfg.mm_obs_threshold in self.thresholds:
                        tk = cfg.mm_obs_tk
                        
        return tk


    def return_predicted_probability(self, thresh_key):
        """ Pull out the predicted value for the requested threshold.
            Performs units checking with observed values.
            
            None is returned if threshold isn't found or model
            doesn't make prediction.
            
        """
        pred_prob = None
        match_status = ""
        
        #Check if a forecast exists for probability
        if self.prediction.probabilities == []:
            return pred_prob, match_status

        #Check each forecast for probability
        for prob_obj in self.prediction.probabilities:
            pred_thresh = {'threshold': prob_obj.threshold,
                'threshold_units': prob_obj.threshold_units}
            tk = objh.threshold_to_key(pred_thresh)
            
            #Check that predicted threshold was applied in the observations
            #If mismatch allowed, will make a new tk that matches observations
            if self.mismatch:
                tk = self.mismatch_thresh_key(pred_thresh)
            
            if tk != thresh_key:
                match_status = "No Matching Threshold"
                continue
            
            pred_prob = prob_obj.probability_value
            match_status = self.sep_match_status[tk]

        return pred_prob, match_status



    def return_predicted_threshold_crossing_time(self, thresh_key):
        """ Pull out the predicted value for the requested threshold.
            
            None is returned if threshold isn't found or model
            doesn't make prediction.
            
        """
        
        predicted = None
        match_status = ""
        
        #Check if a forecast exists for probability
        if self.prediction.threshold_crossings == []:
            return predicted, match_status

        #Check each forecast for probability
        for obj in self.prediction.threshold_crossings:
            pred_thresh = {'threshold': obj.threshold,
                'threshold_units': obj.threshold_units}
            tk = objh.threshold_to_key(pred_thresh)

            #Check that predicted threshold was applied in the observations
            #If mismatch allowed, will make a new tk that matches observations
            if self.mismatch:
                tk = self.mismatch_thresh_key(pred_thresh)
            
            if tk != thresh_key:
                match_status = "No Matching Threshold"
                continue
 
            predicted = obj.crossing_time
            match_status = self.sep_match_status[tk]

        return predicted, match_status



    def return_predicted_start_time(self, thresh_key):
        """ Pull out the predicted value for the requested threshold.
            
            None is returned if threshold isn't found or model
            doesn't make prediction.
            
        """
        predicted = None
        match_status = ""
        
        #Check if a forecast exists for probability
        if self.prediction.event_lengths == []:
            return predicted, match_status

        #Check each forecast for probability
        for obj in self.prediction.event_lengths:
            pred_thresh = {'threshold': obj.threshold,
                'threshold_units': obj.threshold_units}
            tk = objh.threshold_to_key(pred_thresh)
            #Check that predicted threshold was applied in the observations
            if obj.threshold == None:
                continue

            #Check that predicted threshold was applied in the observations
            #If mismatch allowed, will make a new tk that matches observations
            if self.mismatch:
                tk = self.mismatch_thresh_key(pred_thresh)

            if tk != thresh_key:
                match_status = "No Matching Threshold"
                continue

            predicted = obj.start_time
            match_status = self.sep_match_status[tk]

        return predicted, match_status




    def return_predicted_end_time(self, thresh_key):
        """ Pull out the predicted value for the requested threshold.
            
            None is returned if threshold isn't found or model
            doesn't make prediction.
            
        """
        predicted = None
        match_status = ""
        
        #Check if a forecast exists for probability
        if self.prediction.event_lengths == []:
            return predicted, match_status

        #Check each forecast for probability
        for obj in self.prediction.event_lengths:
            pred_thresh = {'threshold': obj.threshold,
                'threshold_units': obj.threshold_units}

            #Check that predicted threshold was applied in the observations
            if obj.threshold == None:
                continue

            tk = objh.threshold_to_key(pred_thresh)
            
            #Check that predicted threshold was applied in the observations
            #If mismatch allowed, will make a new tk that matches observations
            if self.mismatch:
                tk = self.mismatch_thresh_key(pred_thresh)
            
            if tk != thresh_key:
                match_status = "No Matching Threshold"
                continue

            predicted = obj.end_time
            match_status = self.end_time_match_status[tk]

        return predicted, match_status



#    def return_predicted_time_profile(self):
#        """ Pull out the predicted value for the requested threshold.
#            
#            None is returned if threshold isn't found or model
#            doesn't make prediction.
#            
#        """
#        predicted = None
#        match_status = ""
#        
#        #Check if a forecast exists
#        if self.prediction.sep_profile == None:
#            return predicted, match_status
#
#        predicted = self.prediction.sep_profile
#        match_status = self.time_profile_match_status[tk]
#
#        return predicted, match_status



    def return_predicted_fluence(self, thresh_key):
        """ Pull out the predicted value for the requested threshold.
            
            None is returned if threshold isn't found or model
            doesn't make prediction.
            
        """
        predicted = None
        pred_units = None
        match_status = ""

        #Check if a forecast exists for probability
        if self.prediction.fluences == []:
            return predicted, pred_units, match_status

        for obj in self.prediction.fluences:
            pred_thresh = {'threshold': obj.threshold,
                'threshold_units': obj.threshold_units}
            
            #Check that predicted threshold was applied in the observations
            if obj.threshold == None:
                continue

            tk = objh.threshold_to_key(pred_thresh)
            
            #Check that predicted threshold was applied in the observations
            #If mismatch allowed, will make a new tk that matches observations
            if self.mismatch:
                tk = self.mismatch_thresh_key(pred_thresh)
                
            if tk != thresh_key:
                match_status = "No Matching Threshold"
                continue

            match_status = self.sep_match_status[tk]
            predicted = obj.fluence

            pred_units = obj.units
            obs_units = self.observed_fluence[tk].units
            if obs_units != None and pred_units != None:
                if obs_units != pred_units:
                    #Find a conversion factor from the prediction units
                    #to the observation units
                    conv = vunits.calc_conversion_factor(obs_units, pred_units)
                    if conv != None:
                        predicted = predicted * conv
                        pred_units = obs_units
                    else:
                        predicted = None
                        match_status = "Mismatched Units"

        return predicted, pred_units, match_status



    def return_predicted_fluence_spectrum(self, thresh_key):
        """ Pull out the predicted value for the requested threshold.
            
            None is returned if threshold isn't found or model
            doesn't make prediction.
            
        """
        predicted = None
        pred_units = None
        match_status = ""

        #Check if a forecast exists for probability
        if self.prediction.fluence_spectra == []:
            return predicted, pred_units, match_status

        #Check each forecast for probability
        for obj in self.prediction.fluence_spectra:
            pred_thresh = {'threshold': obj.threshold_start,
                'threshold_units': obj.threshold_units}

            #Check that predicted threshold was applied in the observations
            if obj.threshold_start == None:
                continue

            tk = objh.threshold_to_key(pred_thresh)
            
            #Check that predicted threshold was applied in the observations
            #If mismatch allowed, will make a new tk that matches observations
            if self.mismatch:
                tk = self.mismatch_thresh_key(pred_thresh)

            if tk != thresh_key:
                match_status = "No Matching Threshold"
                continue


            match_status = self.sep_match_status[tk]
            predicted = obj.fluence_spectrum

            obs_units = self.observed_fluence_spectrum[tk].fluence_units
            pred_units = obj.fluence_units
            
            if obs_units != None and pred_units != None:
                if obs_units != pred_units:
                    match_status = "Mismatched Units"

        return predicted, pred_units, match_status



    def return_predicted_peak_intensity(self):
        """ Pull out the predicted value.
            
        """
        #Check if prediction exists
        predicted = self.prediction.peak_intensity.intensity
        pred_units = self.prediction.peak_intensity.units
        pred_time = self.prediction.peak_intensity.time
        
        match_status = ""
        if predicted != None:
            match_status = self.peak_intensity_match_status

        #Check units
        obs_units = self.observed_peak_intensity.units
        if obs_units != None and pred_units != None:
            if obs_units != pred_units:
                #Find a conversion factor from the prediction units
                #to the observation units
                conv = vunits.calc_conversion_factor(obs_units, pred_units)
                if conv != None:
                    predicted = predicted * conv
                    pred_units = obs_units

        return predicted, pred_units, pred_time, match_status



    def return_predicted_peak_intensity_max(self):
        """ Pull out the predicted value.
            
        """
        #Check if prediction exists
        predicted = self.prediction.peak_intensity_max.intensity
        pred_units = self.prediction.peak_intensity_max.units
        pred_time = self.prediction.peak_intensity_max.time
        
        match_status = None
        if predicted != None:
            match_status = self.peak_intensity_max_match_status

        #Observed units
        obs_units = self.observed_peak_intensity_max.units
        if obs_units != None and pred_units != None:
            if obs_units != pred_units:
                #Find a conversion factor from the prediction units
                #to the observation units
                conv = vunits.calc_conversion_factor(obs_units, pred_units)
                if conv != None:
                    predicted = predicted * conv
                    pred_units = obs_units

        return predicted, pred_units, pred_time, match_status


