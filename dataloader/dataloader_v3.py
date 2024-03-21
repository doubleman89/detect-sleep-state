
import pandas as pd
import numpy as np
from pathlib import PurePosixPath
from sklearn.model_selection import train_test_split
import tensorflow as tf

class Feature:
    def __init__(self,raw_data, feature_name ='not specified') -> None:
        """
        raw data - 
        serie - manipulated data
        """
        self.raw_data = raw_data
        self.name = feature_name

    def set_ma(self,average_val=[10,5]):
        self.ma = Feature.serie_moving_average(self.raw_data,average_val)
        return self.ma
    
    @classmethod
    def serie_moving_average(cls,serie,average_val=[10,5]):
        MA_serie = np.zeros(serie.shape)
        for j in range(len(average_val)):
            for i in range(average_val[j]):
                MA_serie[i,j] = np.mean(serie[:i+1,j])
                continue
            w = np.repeat(1,average_val[j])/average_val[j]
            MA_serie[average_val[j]-1:,j] = np.convolve(serie[:,j],w,'valid')    
        return MA_serie
    


class Serie:

    encode_list = {
        
        "awake":0,
        "onset":1,
        "sleep":1,
        "wakeup":0,
        "unknown":2,
        # "padded" : 3
    }

    decode_list = {

        0 : "wakeup",
        1 : "onset",
        2 : "unknown"

    }


    

    def __init__(self,serie_id,serie_path,serie_events, optimize = True, feature_engineering = True, **kwargs):
        self.serie_id = serie_id
        """
        raw data - raw data read from csv file
        serie - data for manipulation - updated during feature engineering 
        mask = data after adding segmentation mask 
        slices - data after padding and cutting it into one-day length slices 
        valid_range_tables - used     
        """


        if 'record_interval' in kwargs.keys():
            self.record_interval = kwargs['record_interval']
        else:
            raise ValueError('record interval not found')
        self.daily_steps_total = 24*3600//self.record_interval 


        # check if optimize serie data or load it as it is 
        if optimize == True:
            #memory optimization - load all columns except serie_id - already stored as separate attribute
            self.raw_data = pd.read_csv(serie_path, usecols = lambda x:x != 'series_id')
        else: 
            self.raw_data = pd.read_csv(serie_path)
        
        # create serie for manipulation
        self.serie = self.raw_data

        # feature engineering
        if feature_engineering:
            self.feature_engineering(**kwargs)

        if optimize == True:
            # optimize - overwrites self.serie
            self.optimize_serie(['step'])


        self.serie_length = len(self.serie)
        self.serie_events = serie_events
        self.mask = None 
        self.slices = None 
        self.slice_pads = None 
        self.valid_ranges_table = []
        self.valid_ranges_padding =[]





    def decode_events(self,df):
        df_copy = df.copy()
        # decoded_list = {v: k for k, v in self.__class__.decode_list.items()}
        decoded_list = {k: v for k, v in self.__class__.decode_list.items()}
        df_copy["event"] = df_copy["event"].map(decoded_list)    
        return df_copy

    def optimize_serie(self, optimize_cols = []):
        """optimizes selected columns
        integers - goes to unsigned
        float - downcast to lower format"""
        if len(optimize_cols) > 0:

            series_optimized = self.serie[optimize_cols]
            downcast_dict = {
                'int':'unsigned',
                'float':'float'
                }
            for dtype in downcast_dict.keys():
                series_dtype = series_optimized.select_dtypes(include=[dtype])
                series_dtype = series_dtype.apply(pd.to_numeric,downcast =downcast_dict[dtype])
                try:            
                    self.serie[list(series_dtype.columns)] = series_dtype
                except:
                    pass
    
    def get_daily_step_from_timestamp(self,timestamp):
        def get_data(serie,x):
            timestamp = serie#['timestamp']
            date,time = timestamp.split('T')
            #year,month,day = date.split('-')
            hour,minute,sec = time.split(':')
            sec = sec.split('-')[0]
            hour,minute, sec = int(hour), int(minute), int(sec)
            daily_step = hour*3600//self.record_interval + minute*60//self.record_interval + sec//self.record_interval 

            return daily_step

        def get_daily_step(x):
            def f(serie):
                return get_data(serie,x)
            return f
        
        return timestamp.apply(get_daily_step(self.serie))

    def feature_engineering(self,**kwargs):
        def serie_moving_average(serie,average_val=[10,5]):
            MA_serie = np.zeros(serie.shape)
            for j in range(len(average_val)):
                for i in range(average_val[j]):
                    MA_serie[i,j] = np.mean(serie[:i+1,j])
                    continue
                w = np.repeat(1,average_val[j])/average_val[j]
                MA_serie[average_val[j]-1:,j] = np.convolve(serie[:,j],w,'valid')    
            return MA_serie
        
        if "gradient_difference" in kwargs.keys():
            gradient_diff = kwargs["gradient_difference"]
        else:
            raise ValueError("gradient difference not found")

        if "moving_average_enmo_samples" in kwargs.keys():
            moving_average_enmo_samples = kwargs["moving_average_enmo_samples"]
        else:
            raise ValueError("moving average enmo samples not found")
        
        if "moving_average_gradient_samples" in kwargs.keys():
            moving_average_gradient_samples = kwargs["moving_average_gradient_samples"] 
        else:
            raise ValueError("moving average gradient samples not found")

        
        self.serie['daily_step'] = self.get_daily_step_from_timestamp(self.serie['timestamp'])

        serie_anglez  = self.serie[["anglez"]].to_numpy()
        serie_enmo  = self.serie[["enmo"]].to_numpy()

        ma = serie_moving_average(serie_enmo, average_val=[moving_average_enmo_samples])
        self.serie["enmo_ma"]=ma

        ma = serie_moving_average(serie_anglez, average_val=[moving_average_enmo_samples])
        self.serie["anglez_ma"]=ma
        
        raw_gradient = np.gradient(serie_anglez,gradient_diff,axis = -2)
        ma = serie_moving_average(raw_gradient,average_val=[moving_average_gradient_samples])
        self.serie["gradient_anglez_ma"]= ma

        f1 = np.ediff1d(serie_anglez,5)
        f2 = np.clip(np.power(2, f1/5),-1000,200)
        f3 = serie_moving_average(np.expand_dims(f2,axis  = -1),average_val=[10])
        self.serie["anglez_ediff1d_transformed"] = f3

    def _pad_to(self,x, step_window, padding = 'center'):
        """
        x - data which requires padding
        step_window - amount of steps to which "x" has to be padded"""
        Ni, h,w = x.shape[:-2], x.shape[-2],x.shape[-1]
        pads = (step_window - h)//2
        shape = (*Ni,step_window,w)
        out = np.zeros(shape=shape)
        if padding == 'center':
            if len(Ni) == 0:
                out[pads:(step_window-pads)] = x
            else:
                for ii in np.ndindex(Ni):
                    x_j = 0
                    for jj in range(pads,(step_window-pads)):
                        out_j = (jj,)
                        out[ii+out_j] = x[ii+(x_j,)]
                        x_j+=1
                #out[Ni,pads:(final_length-5)] = x
        elif padding == "end":
            if len(Ni) == 0:
                out[:(step_window-2*pads)] = x
            else:
                for ii in np.ndindex(Ni):
                    x_j = 0
                    for jj in range(0,(step_window-2*pads)):
                        out_j = (jj,)
                        out[ii+out_j] = x[ii+(x_j,)]
                        x_j+=1
                #out[Ni,pads:(final_length-5)] = x
        return out, pads

    def _unpad(self,x, pad, padding = 'center'):
        """
        x - data which requires unpadding
        pad - amount of pads from every side of data (at the beginning and at the end)
        """
        Ni, h,w = x.shape[:-2], x.shape[-2],x.shape[-1]
        if padding == 'center':
            if len(Ni) == 0:
                return x[pad:(h-pad)]
            else:
                shape = (*Ni,h-pad*2,w)
                out = np.zeros(shape=shape)
                for ii in np.ndindex(Ni):
                    out_j= 0 
                    for jj in range(pad,(x.shape[-2]-pad)):
                        out[ii+(out_j,)]= x[ii+(jj,)]
                        out_j +=1
        elif padding == 'end':
            if len(Ni) == 0:
                return x[:(h-2*pad)]
            else:
                shape = (*Ni,h-pad*2,w)
                out = np.zeros(shape=shape)
                for ii in np.ndindex(Ni):
                    out_j= 0 
                    for jj in range(0,(x.shape[-2]-2*pad)):
                        out[ii+(out_j,)]= x[ii+(jj,)]
                        out_j +=1
        return out

    def _detect_change(self, slice):
        for val in self.__class__.encode_list.values():
            if (slice[...,:,-1] == val).all():
                return False
        return True
        

    def create_slices(self,serie, unknown_columns =[], drop_columns = ['daily_step','step','timestamp']):
        """ create slices from the serie 
        serie - serie to slice
        unknown columns - columns to pad with unknown values (most likely it will be a segmentation mask)
        drop_columns - columns to be dropped 
        """
        slices = None

        ds_serie = serie['daily_step']
        #declare max daily step - necessary for further calculations
        max_val = self.daily_steps_total

        # if valid ranges table is empty, fill only one range with full length of serie 
        if len(self.valid_ranges_table) == 0:
            self.valid_ranges_table.append([0,serie.shape[0]-1])


        for valid_range in self.valid_ranges_table:
            # get start and stop range 
            start  = valid_range[0]
            stop = valid_range[1]

            # declare left and right padding values
            left_pad = int(ds_serie[start])
            print(max_val,ds_serie[stop])
            right_pad = max_val -ds_serie[stop]
            self.valid_ranges_padding.append([left_pad,right_pad])
            # drop not necessary columns and get a range
            to_pad_data = serie.drop(columns=drop_columns)[start:stop]
            # create slices to fill with unknown data
            if len(unknown_columns) >= 1:
                unknown_padded_data = to_pad_data[unknown_columns].copy().to_numpy()
                unknown_data = True
            else:
                unknown_data = False
            # create slice to be padded with 0 
            zero_padded_data = to_pad_data.drop(columns=unknown_columns).to_numpy()
            
            
            new_slice = None
            # loop over featuers to be padded 
            for i in range(zero_padded_data.shape[-1]):
                zero_padded_slice = np.expand_dims(np.pad(zero_padded_data[:,i],(left_pad,right_pad),'constant',constant_values=(0,0)),axis =1)

                if new_slice is None:
                    new_slice  = zero_padded_slice
                else:
                    new_slice = np.append(new_slice, zero_padded_slice,axis = -1)

            # loop over featuers to be filled with unknown
            if unknown_data:
                for i in range(unknown_padded_data.shape[-1]): 

                    unknown_padded_slice = unknown_padded_data[:,i]
                    unknown_padded_val = Serie.encode_list["unknown"] # add value from dict here! 
                    unknown_padded_slice = np.append(unknown_padded_slice,[unknown_padded_val for i in range(ds_serie[stop], max_val)])
                    unknown_padded_slice = np.insert(unknown_padded_slice,[0 for i in range(0,left_pad)],[unknown_padded_val for i in range(0,left_pad)])
                    unknown_padded_slice = np.expand_dims(unknown_padded_slice,axis = 1)
                    # add mask 
                    new_slice = np.append(new_slice, unknown_padded_slice,axis = -1)

            if slices is None:
                slices = new_slice
            else:
                try:
                    slices  = np.concatenate((slices,new_slice),axis = 0)
                except ValueError:
                    print(slices.shape,new_slice.shape)
                    raise ValueError

            del new_slice
            del zero_padded_data 


        self.slices = slices

    @staticmethod
    def create_events(serie,y_pred :np.array, padding = "end", only_top_score = True , generate_dummies = False):
        """create df_events 
        input:
        y_pred - score  (predicited vaues )"""
        def detectChange(last_val,current_val):
            if last_val == -1 or current_val == -1:
                return False
            
            # if current_val != Serie.decode_list["onset"] and current_val != Serie.decode_list["wakeup"]:
            #     return False

            return last_val !=current_val
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x)
            return e_x / np.expand_dims(e_x.sum(axis=-1),axis = -1) # only difference        

        # create empty dataframe
        df_events = pd.DataFrame(columns=["series_id","step","event","score"])
        # unpad y_pred
        if serie.slice_pads is None:
            y_pred_unpadded = y_pred
        else:
            y_pred_unpadded = serie._unpad(y_pred,serie.slice_pads,padding = padding)
        # convert from logits to probabilities
        y_pred_unpadded = softmax(y_pred_unpadded)                  
        # create step seg list
        #         - events - segmentation mask
        #         -  score   - predicited vaues for chosen event
        events = np.argmax(y_pred_unpadded,axis = -1,keepdims=True)
        scores = np.max(y_pred_unpadded,axis=-1,keepdims=True)      
        # round to decimals places
        scores = np.round(scores,decimals=1)        

        step = -1
        for slices_num in range(events.shape[0]):
            slice = events[slices_num]
            score = scores[slices_num]
            # flags for filtering only one event per slice
            event_detected = False
            highest_score,event_score = -1.0 , -1.0
            for i in range(events.shape[1]):
                step+=1
                event_val = slice[i][0]
                event_score = score[i][0]

                if i == 0:
                    # do not detect anything during first step
                    continue
                elif not detectChange(slice[i-1][0],event_val):
                    continue
                if only_top_score :
                    event_detected = True
                    # if detected event in slice is not the first or below the highest - continue 
                    if event_score <highest_score and highest_score != -1.0:
                        continue
                    index = step 
                    event = event_val
                    highest_score = event_score
                
                else:
                    print(serie.serie_id,step)
                    df_events.loc[len(df_events.index)] = [serie.serie_id,step,event_val,event_score]                

            
            if event_detected:
                print(serie.serie_id,step)
                df_events.loc[len(df_events.index)] = [serie.serie_id,index,event,highest_score]


        # if nothing was detected: 
        if len(df_events.index) == 0 and generate_dummies:
            df_events.loc[len(df_events.index)] = [serie.serie_id,0,1,0.0]
            df_events.loc[len(df_events.index)] = [serie.serie_id,0,2,0.0]

        # decode events 
        df_events = serie.decode_events(df_events)
        # save as serie events 
        return df_events #,y_pred_unpadded

        
    def get_correct_slices(self):
        # implemented as an interface only 
        raise NotImplementedError 

class TrainSerie(Serie):

    """
    key - offset value +- event e.g. if key =1 and event is in step 290, steps 289-291 are classified as 9 
    value - class value
    """
    multiclass_label_list ={
        1 : 9,
        2 : 8,
        4 : 7,
        8 : 6,
        16 : 5,
        32 : 4,
        64 : 3,
        128 : 2,
        256 : 1,
        }

    def __init__(self, serie_id, serie_path, serie_events,**kwargs):
        super().__init__(serie_id, serie_path, serie_events,**kwargs)
        self.empty_events =  pd.isna(self.serie_events["timestamp"]).all()

    def encode_events(self,df):
        df_copy = df.copy()
        df_copy["event"] = df_copy["event"].map(self.__class__.encode_list)    
        return df_copy
    


    def create_segmentation_mask(self,valid_range):
        """
        df = dataframe with noted nights 
        valid_range = valid range of steps to be taken into account when the previous/next step is NaN (not noted
        segmentation_values - every event has different segmentation value 
        include_Nan = parameter which doesn't take into account Nans, all Nans are implemented as old value, valid range value doesn't apply for this setting

        returns list of segmentation values to be added to dataframe
        )"""
        def set_seg_list_range(step_seg_list,min_range,max_range,encoded_list,event,event_range):
            """sets segmentation with specific range after/before an event
                
                if event_ range ==1 -> sets before an event
                elif == 2 => sets after an event 
            """
            if event_range == 1 :
                if event == encoded_list["wakeup"]:
                    step_seg_list[min_range:max_range] = encoded_list["sleep"]
                elif event == encoded_list["onset"]:
                    step_seg_list[min_range:max_range] = encoded_list["awake"]
            elif event_range == 2: 

                if last_event == encoded_list["wakeup"]:
                    step_seg_list[min_range:max_range] = encoded_list["awake"]
                elif last_event == encoded_list["onset"]:
                    step_seg_list[min_range:max_range] = encoded_list["sleep"] 
            else:
                raise ValueError ("wrong event range")
            
        def set_multi_seg_list(event_no, event):
            """sets multi class segmentation list with different values - depends on the offset value to the event
            """
            
            for offset in sorted(TrainSerie.multiclass_label_list.keys(),reverse=True):
                if event == encoded_list["wakeup"]:
                    event_min = event_no-offset if event_no-offset > 0 else 0 
                    event_max = event_no+offset if event_no+offset+1 < step_multi_seg_list_wakeup.shape[0] else step_multi_seg_list_wakeup.shape[0] 
                    step_multi_seg_list_wakeup[event_min: event_max] = TrainSerie.multiclass_label_list[offset]
                elif event == encoded_list["onset"]:
                    event_min = event_no-offset if event_no-offset > 0 else 0 
                    event_max = event_no+offset if event_no+offset+1 < step_multi_seg_list_onset.shape[0] else step_multi_seg_list_onset.shape[0] 
                    step_multi_seg_list_onset[event_min: event_max] = TrainSerie.multiclass_label_list[offset]


                   
        # encode events with values from encode list
        encoded_list = self.__class__.encode_list
        encoded_events = self.encode_events(self.serie_events)
        # create empty array for events after segmentation
        step_seg_list = np.full(shape =(self.serie_length,),fill_value=encoded_list["unknown"],dtype= np.int64)
        step_multi_seg_list_wakeup = np.full(shape =(self.serie_length,),fill_value=0,dtype= np.int64)
        step_multi_seg_list_onset = np.full(shape =(self.serie_length,),fill_value=0,dtype= np.int64)
        last_step =  -1
        last_timestamp_na = False
        last_event = -1
        first_iter_completed= False
        last_min_valid_step = -1 
        valid_ranges_table = []
        # e.g. to access the `exchange` values as in the OP
        for idx, *row in encoded_events.itertuples():
            event = row[2]
            try:
                step = np.int0(row[3])                
            except:
                step = row[3]
            timestamp = row[4]
            try: 
                timestamp_na = np.isnan(timestamp)
            except:
                timestamp_na = False
            # check if value is accepted 
            if event != encoded_list["wakeup"] and event != encoded_list["onset"]:
                raise ValueError(f"last_event on step {last_step} is not wakeup or onset. Value is {last_event}")

            # check if this is first iteration
            if first_iter_completed == False:
                if timestamp_na:
                    last_timestamp_na = True
                    
                else:
                    min_range= 0
                    max_range=step
                    set_seg_list_range(step_seg_list, min_range,max_range,encoded_list,event,1)
                    set_multi_seg_list(step, event)
                    step_seg_list[step] = event
                    last_min_valid_step = 0
                    
                    
                last_step = step
                last_event = event 
                first_iter_completed = True
                continue
            
            # if range should not be monitored, just continue            
            if timestamp_na and last_timestamp_na :
                last_timestamp_na = True
                
            # if current timestamp shouldn't be monitored, update valid_range on the left side (from last_step (last_step not included))        
            elif timestamp_na and not last_timestamp_na:
                min_range= last_step+1
                max_range=last_step+1+valid_range
                set_seg_list_range(step_seg_list, min_range,max_range,encoded_list,event,1)
                last_timestamp_na = True
                valid_ranges_table.append([last_min_valid_step,max_range])


            # if last timestamp shouldn't be monitored, update valid_range on the right side (up to current_step(current_step included))        
            elif not timestamp_na and last_timestamp_na: 
                min_range= step-valid_range
                max_range=step
                set_seg_list_range(step_seg_list, min_range,max_range,encoded_list,event,2)
                set_multi_seg_list(step, event)
                step_seg_list[step] = event
                last_timestamp_na = False
                last_min_valid_step = min_range
            
            # both timestamps valid - update the range from last_step(last_step not included) to current_step (included)
            else:
                min_range= last_step+1
                max_range=step
                set_seg_list_range(step_seg_list, min_range,max_range,encoded_list,event,2)
                set_multi_seg_list(step, event)
                step_seg_list[step] = event
                last_timestamp_na = False
            
            #update last values to next iteratation
            last_step = step
            last_event = event 

        # save valid range after completion of the all event tables 
        if last_timestamp_na == False:
            valid_ranges_table.append([last_min_valid_step,max_range])
        self.valid_ranges_table = valid_ranges_table

        # create mask
        self.mask = self.serie
        self.mask["event"] = step_seg_list
        self.mask["event_multi_wakeup"] = step_multi_seg_list_wakeup
        self.mask["event_multi_awake"] = step_multi_seg_list_onset

    def create_slices(self, unknown_columns =['event'],drop_columns = ['daily_step','step','timestamp']):
        super().create_slices(self.mask,unknown_columns=unknown_columns, drop_columns=drop_columns)

    
    def get_correct_slices(self):

        # slices_len = self.slices.shape[0]
        # slices_with_zero = np.zeros(shape = (slices_len,),dtype=bool)
        # for i in range(slices_len):
        #     slices_with_zero[i] = 1-(self.__class__.encode_list["unknown"] in self.slices[i,...,-1])
        return self.slices
        


class TestSerie(Serie):

    def __init__(self, serie_id, serie_path, serie_events,**kwargs):
        super().__init__(serie_id, serie_path, serie_events,**kwargs)

    def create_slices(self, unknown_columns =[],drop_columns = ['daily_step','step','timestamp']):
        super().create_slices(self.serie,unknown_columns=unknown_columns, drop_columns=drop_columns)

    def create_events(self, y_pred: np.array):
        self.serie_events, raw_prediction = Serie.create_events(self,y_pred)
        return self.serie_events , raw_prediction
    
    def get_correct_slices(self):
        return self.slices


class Series:
    def __init__(self,data,paths):
        self.data = data
        self.paths = paths
        self.df_events = None
        self.series_ids =  list()
        self.series = {}
        self.series_names = set()  
        self.steps_window, self.valid_steps = self.set_ranges(self.data.slice_length)

    def set_ranges(self,slice_length):
        # every step is recorded within 5s 
        record_interval = self.data.record_interval #[s]
        # slice_length
        slice_length = slice_length
        steps_window = np.int0(np.round(slice_length*60*60/record_interval ))
        time_slices = [(steps_window*(i/4),steps_window*(4-i/4)) for i in range(1,4)]
        valid_steps = np.int0(self.data.valid_range_ifNan*60*60/record_interval )
        
        return steps_window,valid_steps
    

class Train_Series(Series):
    def __init__(self,data,paths):
        self.data = data
        self.paths = paths
        self.df_events = pd.read_csv(self.paths.train_events)
        self.series_ids = list(self.df_events["series_id"].unique())
        self.series = {}
        self.series_names = set()   
        self.steps_window, self.valid_steps = super().set_ranges(self.data.slice_length)
    
    # def set_ranges(self):
    #     steps_window, valid_steps = super().set_ranges()
    #     return steps_window, valid_steps
    
    def get_serie_events(self,serie_id):
        return self.df_events[self.df_events["series_id"]==serie_id]
    
    def createSerie(self,serie_id):
        if serie_id in self.series_names:
            ## TODO - place for logger warning
            return self.series[serie_id]
        serie_filename = serie_id+"."+self.data.series_format
        serie_path = self.paths.train_series / serie_filename
        serie_events = self.get_serie_events(serie_id)
        serie = TrainSerie(serie_id,serie_path,serie_events,gradient_difference = self.data.gradient_difference, moving_average_gradient_samples = self.data.moving_average_gradient_samples, moving_average_enmo_samples = self.data.moving_average_enmo_samples, record_interval = self.data.record_interval)
        return serie
    
    def addSerie(self,serie):
        """do not add empty  serie"""
        if serie.empty_events:
            return False
        self.series[serie.serie_id] = serie
        self.series_names.add(serie.serie_id)
        return True
    
    def createSeries(self):
        # extract all series ids in train series dataset
        p = self.paths.train_series.glob(f'*.{self.data.series_format}')
        series_files = [PurePosixPath(x).stem for x in p if x.is_file()]


        for serie_id in self.series_ids:   
            if serie_id in series_files:
                serie = self.createSerie(serie_id)
                added = self.addSerie(serie)     

                if added == False:
                    continue
                serie.create_segmentation_mask(self.valid_steps)
                serie.create_slices(unknown_columns=['event'])

    def create_events(self, y_pred: np.array):
        """creates events for specific series data"""
        serie_events, raw_prediction = Serie.create_events(self,y_pred)
        return serie_events , raw_prediction

class Test_Series(Series):
    def __init__(self,data,paths):
        super().__init__(data,paths)
        self.steps_window, self.valid_steps = super().set_ranges(self.data.test_slice_length)
    
    # def set_ranges(self):
    #     steps_window, valid_steps = super().set_ranges()
    #     return steps_window, valid_steps

    def createSerie(self,serie_id):
            if serie_id in self.series_names:
                ## TODO - place for logger warning
                return self.series[serie_id]
            serie_filename = serie_id+"."+self.data.series_format
            serie_path = self.paths.test_series / serie_filename
            serie = TestSerie(serie_id,serie_path,None,gradient_difference = self.data.gradient_difference, moving_average_gradient_samples = self.data.moving_average_gradient_samples, moving_average_enmo_samples = self.data.moving_average_enmo_samples, record_interval = self.data.record_interval)
            self.series[serie_id] = serie
            self.series_names.add(serie_id)
            self.series_ids.append(serie_id)
            return serie

    def createSeries(self):
        # extract all series ids in test series dataset
        p = self.paths.test_series.glob(f'*.{self.data.series_format}')
        series_files = [PurePosixPath(x).stem for x in p if x.is_file()]
        for serie_id in series_files:   
            serie = self.createSerie(serie_id)
            serie.create_slices()

    def createSeriesEvents(self,events : dict):
        """
        input: 
        events - dictionary with serie_id as key 
                and values equal to numpy array:
                 - column 1 - segmentation mask
                 - column 2 - score   """
        first_event = False
        series_event = None
        for serie_id in self.series.keys():
            serie_event =self.series[serie_id].create_events(events[serie_id])
            if first_event:
                series_event = serie_event
                first_event = True
                continue
            else:
                series_event = pd.concat([series_event,serie_event])
        
        series_event["row_id"] = range(0,len(series_event))
        self.df_events = series_event
        return self.df_events
    

class Dataset:
    epsilon = 0.000000000000001 

    def __init__(self, train_series, test_series, normalize = True, positional_encoding = True, daily_steps_total = 17280, outputs_no = -3 ) -> None:
        """
        ds- dataset - consists of train series - splits to test and dev sets
        ds_test - dataset - consists of test series - does not have y labels (kaggle competition) """
        def f( x,total) : 
            if x%total == 0:
                return x//total
            raise ValueError(f'{x} divided by {total} should be integer value')
        
        if outputs_no >= 0:
            raise ValueError ("Value should be negative")
        elif type(outputs_no) != int:
            raise TypeError("Value should be int ")
        self.outputs_no = outputs_no
        self.daily_steps_total = daily_steps_total
        self.pos_enc_one_interval = self._positional_encoding_single(self.daily_steps_total,1)

        self.ds, x, self.slices_ids = self._create_dataset_from_slices(train_series)
        self.train_days = f(x,daily_steps_total)
        self.ds_test, x, self.test_slices_ids = self._create_dataset_from_slices(test_series)
        self.test_days = f(x,daily_steps_total)
        self.normalize = normalize
        self.positional_encoding = positional_encoding

        # for windowed datasets     
        self.train = self.ds.copy()
        self.test = self.ds_test.copy()
  
        self.mean = None
        self.std = None 

    
    def make_dataset(self):
        if self.normalize:
            self.normalize_dataset()
        if self.positional_encoding:
            self.add_positional_encoding()

        
        

    @staticmethod
    def positional_encoding_total(max_position,repeats : int, d_model, min_freq=1e-4):
        """
        max_ position - maximum length of the data to encode 
        repeats - when the encoding should be repeated due to repeated intervals (e.g. every day in time-series)
        d_model - dimension of the encoding 
        min_freq - minimum frequency 
        """
        position = np.arange(max_position)
        freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
        pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        for i in range(repeats):
            if i == 0: 
                rep_pos_enc = np.append(pos_enc,pos_enc,axis = 0)
            else:
                rep_pos_enc = np.append(rep_pos_enc,pos_enc,axis = 0)
        
        return rep_pos_enc
    
 
    def _positional_encoding_single(self,max_position,d_model, min_freq=1e-4):
        """
        max_ position - maximum length of the data to encode 
        repeats - when the encoding should be repeated due to repeated intervals (e.g. every day in time-series)
        d_model - dimension of the encoding 
        min_freq - minimum frequency 
        """
        position = np.arange(max_position)
        freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
        pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc
    

    def _positional_encoding_repeats(self, full_repeats : int):
        for i in range(full_repeats):
            rep_pos_enc = np.append(self.pos_enc_one_interval,self.pos_enc_one_interval,axis = 0)
        return rep_pos_enc
    
    def add_positional_encoding(self):
        train_pos_enc = self._positional_encoding_repeats(self.train_days)
        self.ds = np.insert(self.ds,np.arange(self.train_days),train_pos_enc,axis = -1)

        test_pos_enc = self._positional_encoding_repeats(self.test_days)
        self.ds = np.insert(self.ds,np.arange(self.test_days),test_pos_enc,axis = -1)


    def _create_dataset_from_slices(self,series : Series, ):
        """
        return 
        """
        slices_ids = []
        ds_from_slices = None
        for serie_id in series.series.keys(): 
            ms = series.series[serie_id].get_correct_slices()
            if ds_from_slices is None:
                ds_from_slices=ms
                slices_ids.append(((0,ds_from_slices.shape[0]-1),serie_id))
            else:
                try:
                    old_shape = ds_from_slices.shape[0]
                    ds_from_slices = np.concatenate((ds_from_slices,ms),axis = 0)
                    slices_ids.append(((old_shape,ds_from_slices.shape[0]-1),serie_id))
                    print(ds_from_slices.shape, ms.shape,  ds_from_slices.shape[0]/self.daily_steps_total)
                except ms.shape[0] % self.daily_steps_total != 0:
                    print(serie_id)
        
        slices_no = ds_from_slices.shape[0]

        return ds_from_slices, slices_no,slices_ids
    

    def normalize_dataset(self):
        self.train[...,:self.outputs_no] = self._fit_transform(self.train[...,:self.outputs_no])
        self.test = self._transform(self.test)

    def get_windowed_trainset(self,stride,batch_size,shuffle_buffer):
        window_size = self.train.shape[1]
        return Dataset.windowed_dataset(self.train,window_size=window_size,batch_size=batch_size,shuffle_buffer=shuffle_buffer, outputs_no = self.outputs_no)

    def get_windowed_testset(self,batch_size):
        window_size = self.test.shape[1]
        return Dataset.windowed_testset(self.test,window_size,batch_size)

    @staticmethod            
    def windowed_dataset(series,  window_size, batch_size, shuffle_buffer,outputs_no,drop_remainder = True):
        """Generates dataset windows

        Args:
        series (array of float) - contains the values of the time series
        labels - contains 
        window_size (int) - the number of time steps to average
        batch_size (int) - the batch size
        shuffle_buffer(int) - buffer size to use for the shuffle method

        Returns:
        dataset (TF Dataset) - TF Dataset containing time windows
        """
        num_classes = np.max(series[:,:,-1])+1
        multi_class_range = np.max(series[:,:,-2]+1)
        dataset = series.reshape(-1,series.shape[-1])
        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size, shift=window_size, drop_remainder=drop_remainder)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size+1))

        #
        # Create tuples with features and labels 
        dataset = dataset.map(lambda window: (window[:,:outputs_no], 
                                              tf.one_hot(tf.dtypes.cast(window[:,-3],tf.int8),depth=multi_class_range),
                                              tf.one_hot(tf.dtypes.cast(window[:,-2],tf.int8),depth=multi_class_range), 
                                              tf.one_hot(tf.dtypes.cast(window[:,-1],tf.int8),depth=num_classes)))

        # Shuffle the windows
        dataset = dataset.shuffle(shuffle_buffer)
        
        # Create batches of windows
        dataset = dataset.batch(batch_size).prefetch(1)
        
        return dataset

    @staticmethod
    def windowed_testset(series,  window_size,  batch_size,drop_remainder=True):
        """Generates dataset windows

        Args:
        series (array of float) - contains the values of the time series
        labels - contains 
        window_size (int) - the number of time steps to average
        batch_size (int) - the batch size
        shuffle_buffer(int) - buffer size to use for the shuffle method

        Returns:
        dataset (TF Dataset) - TF Dataset containing time windows
        """
        dataset = series.reshape(-1,series.shape[-1])
        print(dataset.shape)

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        print(dataset)
        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size,  drop_remainder=drop_remainder)
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size))

        # Create batches of windows
        dataset = dataset.batch(batch_size).prefetch(1)
        
        return dataset
    def _fit_transform(self,x):
        # normalize through all shapes except the last one
        axis = tuple([i for i in range(len(x.shape)-1)])
        self.mean = x.mean(axis=axis)
        self.std = x.std(axis = axis)
        return self._transform(x)
    
    def _transform(self,x):
        #return (x-self.mean)/(self.__class__.epsilon+self.std)
        return (x)/(self.__class__.epsilon+np.abs(self.std))