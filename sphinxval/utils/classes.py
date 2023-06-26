#Code for classes
#Forecast Class
#Observation Class
#Matching Class
from . import validation_json_handler as vjson
from . import units_handler as vunits

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
            :integrated_intensity: (float) X-ray intensity summed from start to last
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
    def __init__(self, id, fluence, units, uncertainty_low, uncertainty_high):
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
        self.all_clear_boolean = None #All_Clear object
        self.peak_intensity = None #Peak_Intensity object
        self.peak_intensity_max = None #Peak_Intensity object
        self.event_lengths = []
        self.fluences = []
        self.fluence_spectra = []
        self.threshold_crossings = []
        self.probabilities = []
        self.sep_profile = None

        return

    def CheckEnergyChannelValues(self):
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
    def AddTriggersFromDict(self, full_json):
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
                last_data_time, start_time, peak_time, end_time,\
                location, lat, lon, intensity, integrated_intensity, \
                noaa_region = vjson.dict_to_flare(trig['flare'])
                flare = Flare("id", last_data_time, start_time,
                    peak_time, end_time, location, intensity,
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
    def AddInputsFromDict(self,full_json):
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
    def AddForecastsFromDict(self, full_json):
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
        
        #Supporting information
        if dataD != {}:
            if 'species' in dataD:
                self.species = dataD['species']
            
            if 'location' in dataD:
                self.location = dataD['location']
            
            if 'sep_profile' in dataD:
                self.sep_profile = dataD['sep_profile']
            
            if 'prediction_window' in dataD:
                self.prediction_window_start = dataD['prediction_window']['start_time']
                self.prediction_window_end = dataD['prediction_window']['end_time']
            
            if isinstance(self.prediction_window_start, str):
                self.prediction_window_start =\
                    vjson.zulu_to_time(self.prediction_window_start)

            if isinstance(self.prediction_window_end, str):
                self.prediction_window_end =\
                    vjson.zulu_to_time(self.prediction_window_end)

        
        
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
                self.event_lengths.append(Event_Length(start_time,
                    end_time, threshold, threshold_units))
        

        #Load Fluence
        if 'fluences' in dataD:
            for event in dataD['fluences']:
                fluence, units, uncertainty_low, uncertainty_high =\
                    vjson.dict_to_fluence(event)
                self.fluences.append(Fluence("id", fluence, units,
                    uncertainty_low, uncertainty_high))


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


        #Load Probabilities
        if 'probabilities' in dataD:
            for prob in dataD['probabilities']:
                probability_value, uncertainty, threshold,\
                threshold_units = vjson.dict_to_probability(prob)
                self.probabilities.append(Probability(probability_value,
                    uncertainty, threshold, threshold_units))
                    
        return
        


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
        self.all_clear = None #All_Clear object
        self.peak_intensity = None #Peak_Intensity object
        self.peak_intensity_max = None #Peak_Intensity object
        self.event_lengths = []
        self.fluences = []
        self.fluence_spectra = []
        self.threshold_crossings = []
        self.probabilities = []
        self.sep_profile = None

        return


    def CheckEnergyChannelValues(self):
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
    def AddObservationsFromDict(self, full_json):
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
            for event in dataD['fluences']:
                fluence, units, uncertainty_low, uncertainty_high =\
                    vjson.dict_to_fluence(event)
                self.fluences.append(Fluence("id", fluence, units,
                    uncertainty_low, uncertainty_high))


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


        #Load Probabilities
        if 'probabilities' in dataD:
            for prob in dataD['probabilities']:
                probability_value, uncertainty, threshold,\
                threshold_units = vjson.dict_to_probability(prob)
                self.probabilities.append(Probability(probability_value,
                    uncertainty, threshold, threshold_units))
                    
        return



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
        self.issue_time = None
        self.prediction_source = None
        self.prediction_window_start = None
        self.prediction_window_end = None
        self.species = None
        self.location = None

        #MATCHING INFORMATION
        #Observations with observations windows that overlap with
        #the prediction windows - first rough cut at matching
        self.windows_overlap = [] #array of Observation objs
        self.thresholds = [] #all of the thresholds in the observations
        self.threshold_crossed_in_pred_win = [] #filenames of the
            #observations that satisfy the criteria (obj.source)
        self.last_trigger_time = None
        self.last_input_time = None
        self.triggers_before_peak_intensity = []
        self.time_difference_triggers_peak_intensity = [] #hours
        self.inputs_before_peak_intensity = []
        self.time_difference_inputs_peak_intensity = [] #hours
        self.triggers_before_peak_intensity_max = []
        self.time_difference_triggers_peak_intensity_max = [] #hours
        self.inputs_before_peak_intensity = []
        self.time_difference_inputs_peak_intensity_max = [] #hours
        self.observed_threshold_crossing_times = []
        self.triggers_before_threshold_crossing = []
        self.time_difference_triggers_threshold_crossing = [] #hours
        self.inputs_before_threshold_crossing = []
        self.time_difference_inputs_before_threshold_crossing = [] #hours
        
        #PREDICTED AND FORECAST VALUES
        self.predicted_all_clear = None
        self.observed_all_clear = None
        self.all_clear_threshold = None
        self.all_clear_threshold_units = None
        
        return
