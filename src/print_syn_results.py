import pandas as pd
import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.stats

distances_intervals = [1, 2, 3, 4, 100]
angle_intervals = [30, 60, 90, 120, 180]
rt60_intervals = [0.2, 0.4, 0.6, 1.2]

def cal_angle_diff(near_angles, far_angles):
    a0 = near_angles[0]
    diffs = []
    for a1 in far_angles:
        diff = min([np.abs(a1 - a0), np.abs(a1 - a0 - 360), np.abs(a1 - a0 + 360)])
        diffs.append(diff)
    
    return min(diffs)


def return_discrete_index(val, intervals):
    for i, bound in enumerate(intervals):
        if val < bound:
            return i 
    raise ValueError("Valid input val")

def main(args: argparse.Namespace):
    args_path = os.path.join(args.results_dir, 'args.json')
    with open(args_path, 'r') as f:
        args_json = json.load(f)
    info_path = os.path.join(args.results_dir, 'infos.json')
    with open(info_path, 'r') as f:
        infos = json.load(f)

    run_name = os.path.basename(args_json['run_dir'].rstrip('/'))
    print(run_name)
    
    results_csv_path = os.path.join(args.results_dir, 'results.csv')
    results_df = pd.read_csv(results_csv_path)
    
    zero_mask = results_df['n_tgt_speakers'] == 0
    one_mask = results_df['n_tgt_speakers'] == 1
    two_mask = results_df['n_tgt_speakers'] == 2
    

    dis_nears = []
    dis_fars = []
    angle_nears = []
    angle_fars = []

    dis_diff_1spk = []
    angle_diff_1spk = []
    rt60_spk1 = []

    dis_diff_discrete = [[] for i in range(len(distances_intervals))]
    angle_diff_discrete = [[] for i in range(len(angle_intervals))]
    rt60_discrete = [[] for i in range(len(rt60_intervals))]

    for i in range(results_df.shape[0]):
        sample_num = "{:05d}".format(results_df['sample'][i])
        # print(sample_num)
        info = infos[sample_num]
        spatial_info = info["spatial"]
        room_info = info["room"]
        dis_nears.append(spatial_info["dis_near"])
        dis_fars.append(spatial_info["dis_far"])
        angle_nears.append(spatial_info["angle_near"])
        angle_fars.append(spatial_info["angle_far"])
        if one_mask[i]:
            dis_diff = min(spatial_info["dis_far"]) - max(spatial_info["dis_near"]) 
            dis_diff_1spk.append(dis_diff)
            _i = return_discrete_index(dis_diff, distances_intervals)
            dis_diff_discrete[_i].append(results_df["sisdri"][i])

            angle_diff = cal_angle_diff(spatial_info["angle_near"], spatial_info["angle_far"])
            angle_diff_1spk.append(angle_diff)
            _i = return_discrete_index(angle_diff, angle_intervals)
            angle_diff_discrete[_i].append(results_df["sisdri"][i])


            _i = return_discrete_index(room_info["rt60"], rt60_intervals)
            rt60_discrete[_i].append(results_df["sisdri"][i])
            rt60_spk1.append(room_info["rt60"])

            if results_df["sisdri"][i] < 0:
                print(sample_num, results_df["sisdri"][i])
                print(spatial_info)
                print(room_info)
        # print(sample  _num, spatial_info["dis_near"])
    
    ### correlation compution

    results_df["snro"] = results_df["snri"] + results_df["input_snr"]
    results_df["sisdro"] = results_df["sisdri"] + results_df["input_sisdr"]


    if any(zero_mask):
        mean_decay = np.mean(results_df[zero_mask]['decay'])
        std_decay = np.std(results_df[zero_mask]['decay'])
        print(f'Decay: {mean_decay:.02f} +/- {std_decay:.02f}dB')
        
        mean_zero_input_sisdr = np.mean(results_df[zero_mask]['input_sisdr'])
        std_zero_input_sisdr = np.std(results_df[zero_mask]['input_sisdr'])
        print(f'Zero input SI-SDR: {mean_zero_input_sisdr:.02f} +/- {std_zero_input_sisdr:.02f}dB')
    
    if any(one_mask):
        mean_single_sisdri = np.mean(results_df[one_mask]['sisdri'])
        std_single_sisdri = np.std(results_df[one_mask]['sisdri'])
        print(f'Single target SI-SDRi: {mean_single_sisdri:.02f} +/- {std_single_sisdri:.02f}dB')
        
        mean_single_input_sisdr = np.mean(results_df[one_mask]['input_sisdr'])
        std_single_input_sisdr = np.std(results_df[one_mask]['input_sisdr'])
        print(f'Single target input SI-SDR: {mean_single_input_sisdr:.02f} +/- {std_single_input_sisdr:.02f}dB')
    
    if any(two_mask):
        mean_double_sisdri = np.mean(results_df[two_mask]['sisdri'])
        std_double_sisdri = np.std(results_df[two_mask]['sisdri'])    
        print(f'Double target SI-SDRi: {mean_double_sisdri:.02f} +/- {std_double_sisdri:.02f}dB')
        
        mean_double_input_sisdr = np.mean(results_df[two_mask]['input_sisdr'])
        std_double_input_sisdr = np.std(results_df[two_mask]['input_sisdr'])
        print(f'Double target input SI-SDR: {mean_double_input_sisdr:.02f} +/- {std_double_input_sisdr:.02f}dB')
    plt.clf()


    plt.scatter(results_df[one_mask]['input_sisdr'], results_df[one_mask]['sisdri'] + results_df[one_mask]['input_sisdr'], s=0.5)
    plt.plot([min(results_df[one_mask]['input_sisdr']), max(results_df[one_mask]['input_sisdr'])], 
             [min(results_df[one_mask]['input_sisdr']), max(results_df[one_mask]['input_sisdr'])], color='green')
    plt.legend()
    plt.xlabel('Input SI-SDR')
    plt.ylabel('Output SI-SDR')
    plt.ylim([-20, 30])
    plt.savefig( os.path.join(args.results_dir, 'input_vs_output_si_sdr.png') )
    plt.clf()

    plt.scatter(dis_diff_1spk, results_df[one_mask]['sisdri'])
    plt.xlabel('dis_diff')
    plt.xticks(rotation=90)
    plt.ylabel('SI-SDRi')
    plt.savefig(os.path.join(args.results_dir, 'dis_diff.png'))
    plt.clf()

    plt.scatter(angle_diff_1spk, results_df[one_mask]['sisdri'])
    plt.xlabel('angle_diff')
    plt.xticks(rotation=90)
    plt.ylabel('SI-SDRi')
    plt.savefig(os.path.join(args.results_dir, 'angle_diff.png'))
    plt.clf()
    

    plt.scatter(rt60_spk1, results_df[one_mask]['sisdri'])
    plt.xlabel('rt60')
    plt.xticks(rotation=90)
    plt.ylabel('SI-SDRi')
    plt.savefig(os.path.join(args.results_dir, 'rt60.png'))
    plt.clf()


    ### angle 
    y_angle = []
    y_std = []
    x_angle = []
    for i, results in enumerate(angle_diff_discrete):
        print(angle_intervals[i], len(results))
        if len(results) > 0:
            y_angle.append(np.mean(results))
            y_std.append(np.std(results))
            x_angle.append(angle_intervals[i])
        else:
            print("Warning angle!!!!", angle_intervals[i])
    # plt.plot(x_angle, y_angle)
    plt.errorbar(x_angle, y_angle, yerr =y_std )
    plt.xlabel('angle_diff')
    plt.xticks(rotation=90)
    plt.ylabel('SI-SDRi')
    plt.ylim([0, 20])
    plt.savefig(os.path.join(args.results_dir, 'angle_diff2.png'))
    plt.clf()
    ### distance
    y_dis = []
    y_std = []
    x_dis = []
    for i, results in enumerate(dis_diff_discrete):
        print(distances_intervals[i], len(results), np.mean(results))
        if len(results) > 0:
            y_dis.append(np.mean(results))
            y_std.append(np.std(results))
            x_dis.append(i)
        else:
            print("Warning dis!!!!", distances_intervals[i])
    # plt.plot(x_dis, y_dis)
    plt.errorbar(x_dis, y_dis, yerr =y_std )
    plt.xlabel('dis_diff')
    plt.xticks(rotation=90)
    plt.ylabel('SI-SDRi')
    plt.ylim([0, 20])
    plt.savefig(os.path.join(args.results_dir, 'dis_diff2.png'))
    plt.clf()


    ### distance
    y = []
    y_std = []
    x = []
    for i, results in enumerate(rt60_discrete):
        print(rt60_intervals[i], len(results), np.mean(results))
        if len(results) > 0:
            y.append(np.mean(results))
            y_std.append(np.std(results))
            x.append(i)
        else:
            print("Warning dis!!!!", rt60_intervals[i])
    # plt.plot(x_dis, y_dis)
    plt.errorbar(x, y, yerr =y_std )
    plt.xlabel('rt60')
    plt.xticks(rotation=90)
    plt.ylabel('SI-SDRi')
    plt.ylim([0, 20])
    plt.savefig(os.path.join(args.results_dir, 'rt60_2.png'))
    plt.clf()

    ### rt60 

    # for spk in tgt_speakers:
    #     mask = (results_df['tgt_speaker_ids'] == spk) & (results_df['room'] == 'Daogao_cse674')
    #     df = results_df[mask]
    #     mean_single_sisdri = np.mean(df['sisdri'])
    #     std_single_sisdri = np.std(df['sisdri'])
    #     sisdris.append(mean_single_sisdri)

    #     dis = df['tgt_speaker_distances'].apply(lambda x: int(x[1:-1]))
    #     dis = dis.mean()
    #     print(dis, spk)
    #     colors.append((dis/100, dis/100, dis/100))

    # plt.bar([x[2:-2] for x in tgt_speakers], sisdris, color = colors)
    # plt.xlabel('Target Speaker ID')
    # plt.xticks(rotation=90)
    # plt.ylabel('SI-SDRi')
    # plt.savefig(os.path.join(args.results_dir, 'sisdri_vs_tgt_speaker.png'))
    # plt.clf()

    # print(results_df.head())
    # distances = list(results_df['tgt_speaker_distances'].unique())
    # distances = sorted(distances, key=lambda x: int(x[1:-1]))
    # print(distances)
    # sisdris = []
    # counts = []
    # for distance in distances:
    #     df = results_df[results_df['tgt_speaker_distances'] == distance]
    #     mean_single_sisdri = np.mean(df['sisdri'])
    #     std_single_sisdri = np.std(df['sisdri'])
    #     counts.append(df['sisdri'].shape[0])
    #     sisdris.append(mean_single_sisdri)

    # plt.xlabel('Target Speaker Distance')
    # plt.ylabel('SI-SDRi')
    # plt.bar(distances, sisdris)
    # plt.savefig(os.path.join(args.results_dir, 'sisdri_vs_tgt_distance.png'))
    # plt.clf()

    # plt.xlabel('Target Speaker Distance')
    # plt.ylabel('Counts')
    # plt.bar(distances, counts)
    # plt.savefig(os.path.join(args.results_dir, 'freq_vs_tgt_distance.png'))
    # plt.clf()

    # angles = list(results_df['angle'].unique())
    # angles = np.arange(0, 359, 10)
    # print(angles)
    # sisdris = []
    # for angle in angles:
    #     df = results_df[np.abs(results_df['angle'] - angle) <= 5]
    #     mean_single_sisdri = np.mean(df['sisdri'])
    #     std_single_sisdri = np.std(df['sisdri'])
    #     sisdris.append(mean_single_sisdri)

    # plt.xlabel('Target Speaker Angle')
    # plt.ylabel('SI-SDRi')
    # plt.bar(angles, sisdris)
    # plt.savefig(os.path.join(args.results_dir, 'sisdri_vs_tgt_angle.png'))
    # plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str, help='Directory with stored CSV file')
    args = parser.parse_args()

    main(args)
