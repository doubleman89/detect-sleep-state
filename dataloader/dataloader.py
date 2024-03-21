
import pandas as pd
import numpy as np
from pathlib import PurePosixPath
from sklearn.model_selection import train_test_split
import tensorflow as tf



class Serie:

    encode_list = {
        
        "awake":0,
        "onset":1,
        "sleep":1,
        "wakeup":0,
        "unknown":2
    }

    decode_list = {
        # "unknown":0,
        # "awake":1,
        # "onset":2,
        # "sleep":3,
        # "wakeup":4
        0 : "wakeup",
        1 : "onset",
        # 1 : "onset",
        # 0 : "wakeup"
    }
    def __init__(self,serie_id,serie_path,serie_events, optimize = True, feature_engineering = True, **kwargs):
        self.serie_id = serie_id



        # check if optimize serie data or load it as it is 
        if optimize == True:
            #memory optimization - load all columns except serie_id - already stored as separate attribute
            self.serie = pd.read_csv(serie_path, usecols = lambda x:x != 'series_id')
        else: 
            self.serie = pd.read_csv(serie_path)

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
        

        serie_anglez  = self.serie[["anglez"]].to_numpy()
        serie_enmo  = self.serie[["enmo"]].to_numpy()

        ma = serie_moving_average(serie_enmo, average_val=[moving_average_enmo_samples])
        self.serie["enmo_ma"]=ma

        ma = serie_moving_average(serie_anglez, average_val=[moving_average_enmo_samples])
        self.serie["anglez_ma"]=ma
        
        raw_gradient = np.gradient(serie_anglez,gradient_diff,axis = -2)
        ma = serie_moving_average(raw_gradient,average_val=[moving_average_gradient_samples])
        self.serie["gradient_anglez_ma"]= ma

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
        

    def create_slices(self, step_window, drop_columns,serie,limit_slices = False, limit_window = 10 , padding = 'end'):
        """ create slices from the serie 
        step_window - defines how many steps contains every slice
        drop_columns - columns to be dropped 
        serie - serie to be sliced
        limit_slices = limit series slicing only to specific slices sets - every set contains an event change within limit window (func argument)
        """
        # if series length is too short - there will be only one slice which needs to be padded 
        if step_window > len(serie):
            # calculate slice lenght 
            data = serie.drop(columns =drop_columns)
            slices = np.expand_dims(data,axis =0)
            slices, self.slice_pads = self._pad_to(slices,step_window,padding=padding)

        else:    
            # calculate slice columns after dropping
            slice_columns = len(serie.columns) - len(drop_columns)
            slices_num = len(serie)//step_window
            
            # init variables
            slices = np.array([]) 
            new_slices = np.array([])
            slice_pos = 0 
            slice_expected_pos =0 
            wait_for_position = False

            for i in range(slices_num):
                slice = serie[(serie['step'] < step_window+step_window*i) & (serie['step']  >= step_window*i) ]    
                new_slice  =slice.drop(columns =drop_columns).to_numpy()
                
                if limit_slices: 
                    
                    new_slice = np.expand_dims(new_slice,axis =0)
                    if len(new_slices) == 0:
                        new_slices = new_slice
                    elif len(new_slices) <limit_window:
                        new_slices = np.concatenate((new_slices,new_slice),axis = 0)
                    else:
                        new_slices = np.concatenate((new_slices[1:limit_window+1],new_slice),axis = 0)

                    if limit_window > 1:                    
                        slice_pos = slice_pos - 1 
                    
                    # update position if slice is with an event change 
                    if self._detect_change(new_slice):
                        # set random position for slice with detected change
                        if limit_window > 1:                    
                            slice_expected_pos = np.random.randint(0,limit_window-1)
                        else:
                            slice_expected_pos = slice_pos                                             
                        wait_for_position =True
                        slice_pos = new_slices.shape[0]

                    
                    if wait_for_position and slice_pos == slice_expected_pos:
                        if len(slices) ==0 : 
                            slices = new_slices
                        else:
                            slices = np.concatenate((slices,new_slices),axis =0)
                        wait_for_position =False


                else:                        
            # create empty slices array
                    if len(slices) ==0  : 
                        slices = np.zeros(shape=(slices_num,step_window,slice_columns))
                    slices[i,:,:] = new_slice

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



    def __init__(self, serie_id, serie_path, serie_events,**kwargs):
        super().__init__(serie_id, serie_path, serie_events,**kwargs)
        self.empty_events =  pd.isna(self.serie_events["timestamp"]).all()

    def encode_events(self,df):
        df_copy = df.copy()
        df_copy["event"] = df_copy["event"].map(self.__class__.encode_list)    
        return df_copy
    


    def create_segmentation_mask(self,valid_range,drop_columns =[]):
        """
        df = dataframe with noted nights 
        valid_range = valid range of steps to be taken into account when the previous/next step is NaN (not noted
        segmentation_values - every event has different segmentation value 
        
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
                   
        # encode events with values from encode list
        encoded_list = self.__class__.encode_list
        encoded_events = self.encode_events(self.serie_events)
        # create empty array for events after segmentation
        step_seg_list = np.full(shape =(self.serie_length,),fill_value=encoded_list["unknown"],dtype= np.int64)
        last_step =  -1
        last_timestamp_na = False
        last_event = -1
        first_iter_completed= False
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
                    min_range= step-valid_range
                    max_range=step
                    set_seg_list_range(step_seg_list, min_range,max_range,encoded_list,event,1)
                    step_seg_list[step] = event
                    
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

            # if last timestamp shouldn't be monitored, update valid_range on the right side (up to current_step(current_step included))        
            elif not timestamp_na and last_timestamp_na: 
                min_range= step-valid_range
                max_range=step
                set_seg_list_range(step_seg_list, min_range,max_range,encoded_list,event,2)
                step_seg_list[step] = event
                last_timestamp_na = False
            
            # both timestamps valid - update the range from last_step(last_step not included) to current_step (included)
            else:
                min_range= last_step+1
                max_range=step
                set_seg_list_range(step_seg_list, min_range,max_range,encoded_list,event,2)
                step_seg_list[step] = event
                last_timestamp_na = False
            
            #update last values to next iteratation
            last_step = step
            last_event = event 

        # create mask
        #self.mask = self.serie[["step","anglez","enmo"]]
        self.mask = self.serie.drop(columns =drop_columns)
        self.mask["event"] = step_seg_list

    def create_slices(self, time_window, drop_columns,limit_slices=True,limit_window=3):
        super().create_slices(time_window,drop_columns,self.mask,limit_slices=limit_slices,limit_window=limit_window)

    
    def get_correct_slices(self):

        slices_len = self.slices.shape[0]
        slices_with_zero = np.zeros(shape = (slices_len,),dtype=bool)
        for i in range(slices_len):
            slices_with_zero[i] = 1-(self.__class__.encode_list["unknown"] in self.slices[i,...,-1])
        return self.slices[slices_with_zero]
        


class TestSerie(Serie):

    def __init__(self, serie_id, serie_path, serie_events,**kwargs):
        super().__init__(serie_id, serie_path, serie_events,**kwargs)

    def create_slices(self, time_window, drop_columns):
        super().create_slices(time_window,drop_columns,self.serie)

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
        serie = TrainSerie(serie_id,serie_path,serie_events,gradient_difference = self.data.gradient_difference, moving_average_gradient_samples = self.data.moving_average_gradient_samples, moving_average_enmo_samples = self.data.moving_average_enmo_samples  )
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
                serie.create_segmentation_mask(self.valid_steps,["timestamp"])
                serie.create_slices(self.steps_window,["step"],limit_slices=self.data.limit_slices, limit_window=self.data.limit_window)

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
            serie = TestSerie(serie_id,serie_path,None,gradient_difference = self.data.gradient_difference, moving_average_gradient_samples = self.data.moving_average_gradient_samples, moving_average_enmo_samples = self.data.moving_average_enmo_samples)
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
            serie.create_slices(self.steps_window,["step","timestamp"])

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

    def __init__(self, train_series, test_series, normalize = True, shuffle = False) -> None:
        """
        ds- dataset - consists of train series - splits to test and dev sets
        ds_test - dataset - consists of test series - does not have y labels (kaggle competition) """
        self.ds = self._create_dataset_from_slices(train_series, shuffle= shuffle)
        self.ds_test = self._create_dataset_from_slices(test_series)
        self.normalize = normalize
        # for normal split 
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.X_test = None
        self.y_test = None
        # for windowed datasets     
        self.train = None
        self.test = None
        self.dev = None

        self.mean = None
        self.std = None 
        self.slices_ids = [] # to memory specific slices range to specific serie_id

    
    def _create_dataset_from_slices(self,series : Train_Series, shuffle = False):
        ds_from_slices = None
        for serie_id in series.series.keys(): 
            ms = series.series[serie_id].get_correct_slices()
            if ds_from_slices is None:
                ds_from_slices=ms
                self.slices_ids  = [(0,ds_from_slices.shape[0]-1),serie_id]
            else:
                try:
                    old_shape = ds_from_slices.shape[0]
                    ds_from_slices = np.concatenate((ds_from_slices,ms),axis = 0)
                    self.slices_ids.append((old_shape,ds_from_slices.shape[0]-1),serie_id)
                except:
                    pass
        if shuffle:
            np.random.shuffle(ds_from_slices)
            self.slices_ids = [] # memory not longer valid 
        return ds_from_slices
    

    def split_dataset(self, train = 1.0, dev = 0.0, window = False):
        if window == False:
            X = self.ds[...,:-1]
            y = self.ds[...,-1]
            # add additional axis to match the shapes 
            if len(X.shape) != len(y.shape):
                y=y[...,np.newaxis]
            if dev != 0.0:          
                self.X_train, self.X_dev, self.y_train, self.y_dev, = train_test_split(X, y, test_size=dev)          
            else: 
                self.X_train = X
                self.y_train = y
            
            self.X_test = self.ds_test

            if self.normalize: 
                self.X_train = self._fit_transform(self.X_train)
                self.X_test = self._transform(self.X_test)
                if dev != 0.0:
                    self.X_dev = self._transform(self.X_dev)
        else:
            self.train = self.ds
            self.test = self.ds_test
            if self.normalize: 
                self.train[...,:-1] = self._fit_transform(self.train[...,:-1])
                self.test = self._transform(self.test)

    def get_windowed_trainset(self,stride,batch_size,shuffle_buffer):
        window_size = self.train.shape[1]
        return Dataset.windowed_dataset(self.train,window_size=window_size,stride=stride,batch_size=batch_size,shuffle_buffer=shuffle_buffer)

    def get_windowed_testset(self,batch_size):
        window_size = self.test.shape[1]
        return Dataset.windowed_trainset(self.test,window_size,batch_size)

    @staticmethod            
    def windowed_dataset(series,  window_size, stride, batch_size, shuffle_buffer,drop_remainder = True):
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
        dataset = series.reshape(-1,series.shape[-1])
        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        # Window the data but only take those with the specified size
        #dataset = dataset.window(window_size, shift=stride, drop_remainder=drop_remainder)
        dataset = dataset.window(window_size+1, shift=stride, drop_remainder=drop_remainder)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size+1))

        #
        # Create tuples with features and labels 
        #dataset = dataset.map(lambda window: (window[:,:-1], tf.one_hot(tf.dtypes.cast(window[:,-1],tf.int8),depth=num_classes)))
        dataset = dataset.map(lambda window: (window[:-1,:-1], tf.one_hot(tf.dtypes.cast(window[-1:,-1],tf.int8),depth=num_classes)))
        #dataset = dataset.map(lambda window: (window[:,:-1], window[:,-1]))

        # Shuffle the windows
        dataset = dataset.shuffle(shuffle_buffer)
        
        # Create batches of windows
        dataset = dataset.batch(batch_size).prefetch(1)
        
        return dataset

    @staticmethod
    def windowed_trainset(series,  window_size,  batch_size,drop_remainder=True):
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
        dataset = series.reshape(-1,series.shape[-1])
        print(dataset.shape)

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        print(dataset)
        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size,  drop_remainder=drop_remainder)
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size))
        # Create tuples with features and labels 
        #dataset = dataset.map(lambda window: (window[:,:-1], tf.one_hot(tf.dtypes.cast(window[:,-1],tf.int8),depth=num_classes)))
        #dataset = dataset.map(lambda window: (window[:,:]))
        
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