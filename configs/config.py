CFG = {
      "paths":{
        "train_events": "./dataloader/datasets/v1/train-events/train_events.csv",
        "train_series": "./dataloader/datasets/v1/train-series/",
        "test_series":"./dataloader/datasets/v1/test-series/"
      },

      "data": {
        "series_format":   "csv",
        "record_interval": 5,  # every step recorded in interval of x [s]
        "slice_length" :75/360, # in [h] 8/36 previously = 160 steps
        "valid_range_ifNan" : 1, # range of time to consider as valid if Nan was detected (before/after) in[h]
        "clean_data" : True,
        "limit_slices" : True,
        "limit_window": 10
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epoches": 20,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}