CFG = {
      "paths":{
        "train_events": "./dataloader/datasets/v1/train-events/train_events.csv",
        "train_series": "./dataloader/datasets/v1/train-series/",
        "test_series":"./dataloader/datasets/v1/test-series/"
      },

      "data": {
        "series_format":   "csv",
        "record_interval": 5,  # every step recorded in interval of x [s]
        "slice_length" :80/360, #8/36, # in [h] 75/360 previously = 150 steps
        "test_slice_length":5/360, # should be equal to slice_length if windowed dataset is not used
        "valid_range_ifNan" : 1, # range of time to consider as valid if Nan was detected (before/after) in[h]
        "clean_data" : True,
        "limit_slices" : True,
        "limit_window": 3,
        "feature_engineering" : True,
        "gradient_difference" : 1,
        "moving_average_gradient_samples" : 3,
        "moving_average_enmo_samples" : 10,
        "shuffle_Dataset" : False,
        "normalize" : True
    },
    "train": {
        "batch_size": 64,
        "shuffle_buffer_size": 1500,
        "stride" : 1,
        "epoches": 30,
    },
    "model": {
        "output": 2
    }
}