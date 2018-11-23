
#import os
import glob
import cv2
import numpy as np
import random
import pandas as pd
from pandas.tools.plotting import table
import matplotlib.pyplot as plt
from os import walk

#import seaborn as sns; sns.set()
#import scipy.misc

class Events(object):
    """
    Temporal Difference events.
    data: a NumPy Record Array with the following named fields
        x: pixel x coordinate, unsigned 16bit int
        y: pixel y coordinate, unsigned 16bit int
        p: polarity value, boolean. False=off, True=on
        ts: timestamp in microseconds, unsigned 64bit int
    width: The width of the frame. Default = 304.
    height: The height of the frame. Default = 240.
    """
    def __init__(self, num_events, width=304, height=240):
        """num_spikes: number of events this instance will initially contain"""
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.uint64)], shape=(num_events))
        self.width = width
        self.height = height

    def show_em(self):
        """Displays the EM events (grayscale ATIS events)"""
        frame_length = 24e3
        t_max = self.data.ts[-1]
        frame_start = self.data[0].ts
        frame_end = self.data[0].ts + frame_length
        max_val = 1.16e5
        min_val = 1.74e3
        val_range = max_val - min_val

        thr = np.rec.array(None, dtype=[('valid', np.bool_), ('low', np.uint64), ('high', np.uint64)], shape=(self.height, self.width))
        thr.valid.fill(False)
        thr.low.fill(frame_start)
        thr.high.fill(0)

        def show_em_frame(frame_data):
            """Prepare and show a single frame of em data to be shown"""
            for datum in np.nditer(frame_data):
                ts_val = datum['ts'].item(0)
                thr_data = thr[datum['y'].item(0), datum['x'].item(0)]

                if datum['p'].item(0) == 0:
                    thr_data.valid = 1
                    thr_data.low = ts_val
                elif thr_data.valid == 1:
                    thr_data.valid = 0
                    thr_data.high = ts_val - thr_data.low

            img = 255 * (1 - (thr.high - min_val) / (val_range))
            #thr_h = cv2.adaptiveThreshold(thr_h, 255,
            #cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
            img = np.piecewise(img, [img <= 0, (img > 0) & (img < 255), img >= 255], [0, lambda x: x, 255])
            img = img.astype('uint8')
            cv2.imshow('img', img)
            cv2.waitKey(1)

        while frame_start < t_max:
            #with timer.Timer() as em_playback_timer:
            frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
            show_em_frame(frame_data)
            frame_start = frame_end + 1
            frame_end += frame_length + 1
            #print 'showing em frame took %s seconds' %em_playback_timer.secs

        cv2.destroyAllWindows()
        return

    def show_td(self, wait_delay=1):
        """Displays the TD events (change detection ATIS or DVS events)
        waitDelay: milliseconds
        """
        frame_length = 24e3
        t_max = self.data.ts[-1]
        frame_start = self.data[0].ts
        frame_end = self.data[0].ts + frame_length
        td_img = np.ones((self.height, self.width), dtype=np.uint8)
        while frame_start < t_max:
            frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]

            if frame_data.size > 0:
                td_img.fill(128)

                #with timer.Timer() as em_playback_timer:
                for datum in np.nditer(frame_data):
                    td_img[datum['y'].item(0), datum['x'].item(0)] = datum['p'].item(0)
                #print 'prepare td frame by iterating events took %s seconds'
                #%em_playback_timer.secs

                td_img = np.piecewise(td_img, [td_img == 0, td_img == 1, td_img == 128], [0, 255, 128])
                cv2.imshow('img', td_img)
                cv2.waitKey(wait_delay)

            frame_start = frame_end + 1
            frame_end = frame_end + frame_length + 1

        cv2.destroyAllWindows()
        return

    def sort_order(self):
        """Generate data sorted by ascending ts
        Does not modify instance data
        Will look through the struct events, and sort all events by the field 'ts'.
        In other words, it will ensure events_out.ts is monotonically increasing,
        which is useful when combining events from multiple recordings.
        """
        #chose mergesort because it is a stable sort, at the expense of more
        #memory usage
        events_out = np.sort(self.data, order='ts', kind='mergesort')
        return events_out

    def extract_roi(self, top_left, size, is_normalize=False):
        """Extract Region of Interest
        Does not modify instance data
        Generates a set of td_events which fall into a rectangular region of interest with
        top left corner at 'top_left' and size 'size'
        top_left: [x: int, y: int]
        size: [width, height]
        is_normalize: bool. If True, x and y values will be normalized to the cropped region
        """
        min_x = top_left[0]
        min_y = top_left[1]
        max_x = size[0] + min_x
        max_y = size[1] + min_y
        extracted_data = self.data[(self.data.x >= min_x) & (self.data.x < max_x) & (self.data.y >= min_y) & (self.data.y < max_y)]

        if is_normalize:
            self.width = size[0]
            self.height = size[1]
            extracted_data = np.copy(extracted_data)
            extracted_data = extracted_data.view(np.recarray)
            extracted_data.x -= min_x
            extracted_data.y -= min_y

        return extracted_data

    def apply_refraction(self, us_time):
        """Implements a refractory period for each pixel.
        Does not modify instance data
        In other words, if an event occurs within 'us_time' microseconds of
        a previous event at the same pixel, then the second event is removed
        us_time: time in microseconds
        """
        t0 = np.ones((self.width, self.height)) - us_time - 1
        valid_indices = np.ones(len(self.data), np.bool_)

        #with timer.Timer() as ref_timer:
        i = 0
        for datum in np.nditer(self.data):
            datum_ts = datum['ts'].item(0)
            datum_x = datum['x'].item(0)
            datum_y = datum['y'].item(0)
            if datum_ts - t0[datum_x, datum_y] < us_time:
                valid_indices[i] = 0
            else:
                t0[datum_x, datum_y] = datum_ts

            i += 1
        #print 'Refraction took %s seconds' % ref_timer.secs

        return self.data[valid_indices.astype('bool')]

    def write_j_aer(self, filename):
        """
        writes the td events in 'td_events' to a file specified by 'filename'
        which is compatible with the jAER framework.
        To view these events in jAER, make sure to select the DAVIS640 sensor.
        """
        import time
        y = 479 - self.data.y
        #y = td_events.y
        y_shift = 22 + 32

        x = 639 - self.data.x
        #x = td_events.x
        x_shift = 12 + 32

        p = self.data.p + 1
        p_shift = 11 + 32

        ts_shift = 0

        y_final = y.astype(dtype=np.uint64) << y_shift
        x_final = x.astype(dtype=np.uint64) << x_shift
        p_final = p.astype(dtype=np.uint64) << p_shift
        ts_final = self.data.ts.astype(dtype=np.uint64) << ts_shift
        vector_all = np.array(y_final + x_final + p_final + ts_final, dtype=np.uint64)
        aedat_file = open(filename, 'wb')

        version = '2.0'
        aedat_file.write('#!AER-DAT' + version + '\r\n')
        aedat_file.write('# This is a raw AE data file - do not edit\r\n')
        aedat_file.write \
            ('# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n')
        aedat_file.write('# Timestamps tick is 1 us\r\n')
        aedat_file.write('# created ' + time.strftime("%d/%m/%Y") \
            + ' ' + time.strftime("%H:%M:%S") \
            + ' by the Python function "write2jAER"\r\n')
        aedat_file.write \
            ('# This function fakes the format of DAVIS640 to allow for the full ATIS address space to be used (304x240)\r\n')
        ##aedat_file.write(vector_all.astype(dtype='>u8').tostring())
        to_write = bytearray(vector_all[::-1])
        to_write.reverse()
        aedat_file.write(to_write)
        #aedat_file.write(vector_all)
        #vector_all.tofile(aedat_file)
        aedat_file.close()


def load_NMNIST(data_file_path):
    'loads up binary data files'
    with open("mybinfile", "rb") as f:
        byte = f.read(1)
        while byte != "":
            # Do stuff with byte.
            byte = f.read(1)


def read_dataset(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7 #bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    #Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    #Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    td = Events(td_indices.size, 34, 34)
    td.data.x = all_x[td_indices]
    td.width = td.data.x.max() + 1
    td.data.y = all_y[td_indices]
    td.height = td.data.y.max() + 1
    td.data.ts = all_ts[td_indices]
    td.data.p = all_p[td_indices]
    return td


if __name__ == '__main__':

    #data_set_path = 'C:/Users/millir/Desktop/4th Year Project/LowPowerActionRecognition/CNN/datasets/Train/Train/0/00002.bin'
    #td = read_dataset(data_set_path)

    #creates a list f that contains local directory
    f = []
    filename1=0
    for (dirpath, dirnames, filename2) in walk('C:/Users/millir/Desktop/4th Year Project/LowPowerActionRecognition/CNN/datasets/Train/Train/'+str(filename1)):
        f.extend(filename2)
        break

        for filename2 in f:

            test+=1
            td = read_dataset('C:/Users/millir/Desktop/4th Year Project/LowPowerActionRecognition/CNN/datasets/Train/Train/'+str(filename1)+'/'+filename2)
            #print(td.height)
            #test+=1
            t_min = np.min(td.data.ts) + 200000
            t_max = np.max(td.data.ts) - 200000
            # Randomly selected a t_r value b/w the range. We only interested in the values stored in the period t_r and t_ru
            t_r = random.uniform(t_min,t_max)
            t_ru = t_r + 100000
            #print(t_r,t_ru)
            # index_need and index_need2 are the lower and upper indexes
            i_low = np.argmin(np.abs(td.data.ts - t_r))
            i_up = np.argmin(np.abs(td.data.ts - t_ru))

            #print(len(range(i_low,i_up)))
            x=[]
            y=[]
            p=[]
            for index in range(i_low,i_up):
                x.append(td.data.x[index])
                y.append(td.data.y[index])
                p.append(td.data.p[index])

            for (i,item) in enumerate(p):
                if item == True:
                    p[i] = 1
                else:
                    p[i] = -1


             #create a dataframe x,y,p out of the 3 lists
            df=pd.DataFrame({'x':x,'y':y,'p':p})

            #sort by x and y
            df1=df.sort_values(['x','y'])
            pd.set_option('display.max_rows',None)

            #group by x and y, sum polarities for each x-y coordinate, and display x,y,p and summed p
            df1['sum_p'] = df1.groupby(['x','y'])['p'].transform(sum)

            #drop old p, drop duplicates, and reset dataframe index
            df2=df1.drop(['p'],axis=1).drop_duplicates(subset=['x', 'y','sum_p']).reset_index(drop=True)

            #check ranges of x,y and sum_p #sanitycheck
            #print(max(df2['x']))
            #print(max(df2['y']))
            #print(min(df2['sum_p']))
            #print(max(df2['sum_p']))

            #prepopulate 34x34 matrix with zeros
            A = np.zeros(shape=(34,34))

            #convert dataframe to np array B to allow easier indexing
            B=df2.values

            #fit B into A so that A(x,y)=p
            for row in B:
                A[row[1]][row[0]]=row[2]

            #convert back to dataframe
            pd.set_option('display.max_rows',None)
            dfA=pd.DataFrame(A)
            print(dfA)

            #visualize
            plt.imshow(dfA, cmap='hot', interpolation='nearest')
            plt.savefig("mytable"+filenames+".png")

        if test >= len(f)+1:
            print("iterations "+str(test)+"/"+str(len(f)))
            filename1 += 1
            test=0

            if filename1 == 10:
                break
        if test == 3:
            print("hello dawgz")
            print("iterations "+str(test)+"/"+str(len(f)))
            break
        #print(len(f))
        #print(test)

        #for i in range(10)
            #string to list
            #data_set_path = to string 'C:/Users/millir/Desktop/4th Year Project/LowPowerActionRecognition/CNN/datasets/Train/Train/i/00002.bin'

        #address_list = list('C:/Users/millir/Desktop/4th Year Project/LowPowerActionRecognition/CNN/datasets/Train/Train/0/00002.bin')[-11:-10]
        #data_set_path = ''.join(address_list)
        #td = read_dataset(data_set_path)
        #print(td.width)
