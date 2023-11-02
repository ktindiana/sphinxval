from . import keys


def process_numbers(injson):
    """ Go through all the numerical entries in the json and
        ensure they are in float variables.
        
        This subroutine uses set_json_value_by_index() in
        validation_json_handler.py. BE CAREFUL AND NOTE that this
        subroutine changes the value of the entries inherently in
        the input json (injson) itself. Once injson is passed
        into this routine, it will be modified even if
        nothing is returned as output.
        
        INPUTS:
        
        :injson: (json dictionary) single forecast or observation json
        
        OUTPUTS
        
        :outjson:(json dictionary) same json but with units fields modified
        
    """
    #all of the possible units entries in a json block
    keys_all_floats = keys.id_all_floats
    keys_all_arrays = keys.id_all_arrays

    nblocks = return_nforecasts(injson)
    narr = -1
    for i in range(nblocks):
        for key in keys_all_floats:
            key_chain = keys.get_key_chain(key)
            
            #Check for values stored in arrays
            if any(x in key_chain for x in keys_all_arrays):
                for arr_key in keys_all_arrays:
                    if arr_key in key_chain:
                        narr = len(return_json_value_by_index(injson,arr_key,i))
                        for j in range(narr):
                            float_val = return_json_value_by_index(injson,key,i,j)
                            if float_val == vars.errval: continue
                            if isinstance(float_val,str) and float_val != ""\
                                and float_val != None:
                                float_val = float(float_val)
                                check =\
                                    set_json_value_by_index(float_val,injson,key,i,j)
            #Not an array-based field
            else:
                float_val = return_json_value_by_index(injson,key,i)
                if float_val == vars.errval: continue
                if isinstance(float_val,str) and float_val != "" and float_val != None:
                    float_val = float(float_val)
                    check = set_json_value_by_index(float_val,injson,key,i)
            
    return injson

