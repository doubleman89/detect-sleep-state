# detect-sleep-state
Child Mind Institute - Detect Sleep States Detect sleep onset and wake from wrist-worn accelerometer data

Project based on dataset from competition - https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states 
- [x] 1. Train series were taken from parquet file in chunks and separated into csv files with following pattern "serie_id".csv
- [x] 2. Train events data were changed into segmenatation mask (0 - awake/wakeup, 1 - onset/sleep) which was included with train series.
- [x] 3. If sleep event in train event was not recognized - segmentation mask is assumed as unkonwn (-1) which will be ignored during training
- [x] 4. First idea is to use U-net model with Conv1D. First learnings made 
- [ ] 5. Train output mask needs to be converted into events 
- [ ] 6. Error analysis, fine tuning etc. 
