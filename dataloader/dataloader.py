import pandas as pd
import numpy as np
from pathlib import PurePosixPath
from sklearn.model_selection import train_test_split



class Serie:

    encode_list = {
        "unknown":-1,
        "awake":0,
        "onset":1,
        "sleep":1,
        "wakeup":0
    }

    decode_list = {
        "wakeup":0,
        "onset":1
    }
    def __init__(self,serie_id,serie_path,serie_events=None,augmentation =False):
        self.serie_id = serie_id
        self.serie = pd.read_csv(serie_path)
        self.serie_length = len(self.serie)
        self.serie_events = serie_events
        self.mask = None 
        self.slices = None 
        self.augmentation = augmentation

    def decode_events(self,df):
        df_copy = df.copy()
        decoded_list = {v: k for k, v in self.__class__.decode_list.items()}
        df_copy["event"] = df_copy["event"].map(decoded_list)    
        return df_copy

    def create_slices(self, time_window, drop_columns,serie):
        # calculate slice lenght
        slice_len = np.int64(len(serie)/time_window)
        # calculate slice columns after dropping
        slice_columns = len(serie.columns) - len(drop_columns)
        # create empty slices array
        slices = np.zeros(shape=(slice_len,time_window,slice_columns))

        slices_num = (len(serie)//time_window)
        # aug_offsets = 0
        # if self.augmentation:
        #     aug_offsets = np.random.random_integers(low=-time_window+1,high=time_window-1,size=slices_num)

        for i in range(slices_num):
            slice = serie[(serie['step'] < time_window+time_window*i) & (serie['step']  >= time_window*i) ]    
            new_slice  =slice.drop(columns =drop_columns)
            slices[i,:,:] = new_slice.to_numpy()

        self.slices = slices
        

class TrainSerie(Serie):



    def __init__(self, serie_id, serie_path, serie_events, augmentation=False):
        super().__init__(serie_id, serie_path, serie_events, augmentation)


    def encode_events(self,df):
        df_copy = df.copy()
        df_copy["event"] = df_copy["event"].map(self.__class__.encode_list)    
        return df_copy
    


    def create_segmentation_mask(self,valid_range):
        """
        df = dataframe with noted nights 
        valid_range = valid range of steps to be taken into account when the previous/next step is NaN (not noted
        segmentation_values - every event has different segmentation value 
        
        returns list of segmentation values to be added to dataframe
        )"""
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
                step = np.int64(row[3])
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
                    
                    if event == encoded_list["wakeup"]:
                        step_seg_list[min_range:max_range] = encoded_list["sleep"]
                    elif event == encoded_list["onset"]:
                        step_seg_list[min_range:max_range] = encoded_list["awake"]

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
                
                if last_event == encoded_list["wakeup"]:
                    step_seg_list[min_range:max_range] = encoded_list["awake"]
                elif last_event == encoded_list["onset"]:
                    step_seg_list[min_range:max_range] = encoded_list["sleep"] 
                
                last_timestamp_na = True

            # if last timestamp shouldn't be monitored, update valid_range on the right side (up to current_step(current_step included))        
            elif not timestamp_na and last_timestamp_na: 
                min_range= step-valid_range
                max_range=step
                
                if event == encoded_list["wakeup"]:
                    step_seg_list[min_range:max_range] = encoded_list["sleep"]
                elif event == encoded_list["onset"]:
                    step_seg_list[min_range:max_range] = encoded_list["awake"]
                
                step_seg_list[step] = event
                last_timestamp_na = False
            
            # both timestamps valid - update the range from last_step(last_step not included) to current_step (included)
            else:
                min_range= last_step+1
                max_range=step
                
                if event == encoded_list["wakeup"]:
                    step_seg_list[min_range:max_range] = encoded_list["sleep"]
                elif event == encoded_list["onset"]:
                    step_seg_list[min_range:max_range] = encoded_list["awake"]
                
                step_seg_list[step] = event
                last_timestamp_na = False
            
            #update last values to next iteratation
            last_step = step
            last_event = event 

        # create mask
        self.mask = self.serie.copy()
        self.mask["event"] = step_seg_list

    def create_slices(self, time_window, drop_columns):
        super().create_slices(time_window,drop_columns,self.mask)

    
    def get_correct_slices(self):
        # correct_slices = self.mask_slices.copy()
        # deleted_indexes = []
        # for i in range(correct_slices.shape[0]):
        #     if self.__class__.encode_list["unknown"] in  correct_slices[i,1]:
        #         deleted_indexes.append[i]
        # return correct_slices.delete(deleted_indexes,axis = 0)
        slices_len = self.slices.shape[0]
        slices_with_zero = np.zeros(shape = (slices_len,),dtype=bool)
        for i in range(slices_len):
            slices_with_zero[i] = 1-(self.__class__.encode_list["unknown"] in self.slices[i,...,-1])
        return self.slices[slices_with_zero]
        


class TestSerie(Serie):

    def __init__(self, serie_id, serie_path, serie_events=None, augmentation=False):
        super().__init__(serie_id, serie_path, serie_events, augmentation)

    def create_slices(self, time_window, drop_columns):
        super().create_slices(time_window,drop_columns,self.serie)


    def create_events(self,step_seg_list : np.array):
        """create df_events 
        input:
        step_seg_list - numpy array:
                 - column 1 - segmentation mask
                 - column 2 - score  (predicited vaues )"""
        def detectChange(last_val,current_val):
            if last_val == -1 or current_val == -1:
                return False

            return last_val !=current_val

        # create empty dataframe
        df_events = pd.DataFrame(columns=["series_id","step","event","score"])
        for i in range(step_seg_list.shape[0]):
            event_val = step_seg_list[i,0]
            event_score = step_seg_list[i,1]
            if i == 0:
                # do not detect anything during first step
                continue
            elif not detectChange(step_seg_list[i-1,0],event_val):
                continue

            df_events.loc[len(df_events.index)] = [self.serie_id,i,event_val,event_score]
        
        # decode events 
        df_events = self.decode_events(df_events)
        # save as serie events 
        self.serie_events = df_events

        return self.serie_events 
    def get_example(self):
        return self.serie[["anglez","enmo"]].to_numpy()

class Series:
    def __init__(self,data,paths):
        self.data = data
        self.paths = paths
        self.df_events = pd.read_csv(self.paths.train_events)
        self.series_ids = list(self.df_events["series_id"].unique())
        self.series = {}
        self.series_names = set()   
        self.steps_window, self.valid_steps = self.set_ranges()
    
    def set_ranges(self):
        # every step is recorded within 5s 
        record_interval = self.data.record_interval #[s]
        # slice_length
        slice_length = self.data.slice_length
        steps_window = np.int64(slice_length*60*60/record_interval )
        time_slices = [(steps_window*(i/4),steps_window*(4-i/4)) for i in range(1,4)]
        valid_steps = np.int64(self.data.valid_range_ifNan*60*60/record_interval )
        
        return steps_window,valid_steps

    def get_serie_events(self,serie_id):
        return self.df_events[self.df_events["series_id"]==serie_id]
    
    def createSerie(self,serie_id):
            if serie_id in self.series_names:
                ## TODO - place for logger warning
                return self.series[serie_id]
            serie_filename = serie_id+"."+self.data.series_format
            serie_path = self.paths.train_series / serie_filename
            serie_events = self.get_serie_events(serie_id)
            serie = TrainSerie(serie_id,serie_path,serie_events)
            self.series[serie_id] = serie
            self.series_names.add(serie_id)
            return serie

    def createSeries(self):
        # extract all series ids in train series dataset
        p = self.paths.train_series.glob(f'*.{self.data.series_format}')
        series_files = [PurePosixPath(x).stem for x in p if x.is_file()]


        for serie_id in self.series_ids:   
            if serie_id in series_files:
                serie = self.createSerie(serie_id)
                serie.create_segmentation_mask(self.valid_steps)
                serie.create_slices(self.steps_window,["series_id","step","timestamp"])



class Test_Series:
    def __init__(self,data,paths):
        self.data = data
        self.paths = paths
        self.df_events = None
        self.series_ids =  None
        self.series = {}
        self.series_names = set()   
        self.steps_window, self.valid_steps = self.set_ranges()
    
    def set_ranges(self):
        # every step is recorded within 5s 
        record_interval = self.data.record_interval #[s]
        # slice_length
        slice_length = self.data.slice_length
        steps_window = np.int64(slice_length*60*60/record_interval )
        time_slices = [(steps_window*(i/4),steps_window*(4-i/4)) for i in range(1,4)]
        valid_steps = np.int64(self.data.valid_range_ifNan*60*60/record_interval )
        
        return steps_window,valid_steps

    def createSerie(self,serie_id):
            if serie_id in self.series_names:
                ## TODO - place for logger warning
                return self.series[serie_id]
            serie_filename = serie_id+"."+self.data.series_format
            serie_path = self.paths.test_series / serie_filename
            serie = TestSerie(serie_id,serie_path)
            self.series[serie_id] = serie
            self.series_names.add(serie_id)
            return serie

    def createSeries(self):
        # extract all series ids in test series dataset
        p = self.paths.test_series.glob(f'*.{self.data.series_format}')
        series_files = [PurePosixPath(x).stem for x in p if x.is_file()]
        for serie_id in series_files:   
            serie = self.createSerie(serie_id)
            serie.create_slices(self.steps_window,["series_id","step","timestamp"])

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
    epsilon = 0.000001 

    def __init__(self, series, normalize = True) -> None:
        self.ds = self._create_dataset_from_slices(series)
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.x_test = None
        self.y_test = None
        self.mean = None
        self.std = None 

    
    def _create_dataset_from_slices(self,series : Series):
        ds_from_slices = None
        for serie_id in series.series: 
            ms = series.series[serie_id].get_correct_slices()
            if ds_from_slices is None:
                ds_from_slices=ms
            else:
                ds_from_slices = np.concatenate((ds_from_slices,ms),axis = 0)
        
        np.random.shuffle(ds_from_slices)
        return ds_from_slices
    

    def split_dataset(self, train = 0.8, dev = 0.0, test = 0.2):
        X = self.ds[...,:-1]
        y = self.ds[...,-1]
        # add additional axis to match the shapes 
        if len(X.shape) != len(y.shape):
            y=y[...,np.newaxis]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=dev+test)          

        if dev != 0.0 and test != 0.0:
            self.X_dev, self.X_dev, self.y_train, self.y_test = train_test_split(self.X_test, self.y_test, test_size=test/(dev+test))

        if self.normalize: 
            self.X_train = self._fit_transform(self.X_train)
            self.X_test = self._transform(self.X_test)
            if self.X_dev is not None:
                self.X_dev = self._transform(self.X_dev)
    
    def _fit_transform(self,x):
        # normalize through all shapes except the last one
        axis = tuple([i for i in range(len(x.shape)-1)])
        self.mean = x.mean(axis=axis)
        self.std = x.std(axis = axis)
        return self._transform(x)
    
    def _transform(self,x):
        return (x-self.mean)/(self.__class__.epsilon+self.std)

def create_dataset_from_slices(series : Series):
    ds_from_slices = None
    for serie_id in series.series: 
        ms = series.series[serie_id].mask_slices
        if ds_from_slices is None:
            ds_from_slices=ms
        else:
            ds_from_slices = np.concatenate((ds_from_slices,ms),axis = 0)
    
    np.random.shuffle(ds_from_slices)
    return ds_from_slices

