import pandas as pd
import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.stats

def main(args: argparse.Namespace):
    args_path = os.path.join(args.results_dir, 'args.json')
    with open(args_path, 'r') as f:
        args_json = json.load(f)
    run_name = os.path.basename(args_json['run_dir'].rstrip('/'))

    results_csv_path = os.path.join(args.results_dir, 'results.csv')
    results_df = pd.read_csv(results_csv_path)

    zero_mask = results_df['n_tgt_speakers'] == 0
    one_mask = results_df['n_tgt_speakers'] == 1
    two_mask = results_df['n_tgt_speakers'] == 2
    non_zero_mask = one_mask | two_mask
    
    
    ### correlation compution

    results_df["snro"] = results_df["snri"] + results_df["input_snr"]
    results_df["sisdro"] = results_df["sisdri"] + results_df["input_sisdr"]
    '''
    x_lists = ["snro", "sisdro", "hubert", "wavlm"]
    y_lists = ["pesq", "stoi"]
    
    for x in x_lists:
         plt.figure()
         for yi in range(len(y_lists)):
    #         ### output
             y = y_lists[yi]
             r = scipy.stats.pearsonr(results_df[x], results_df[y])
             rho = scipy.stats.spearmanr(results_df[x], results_df[y])
             print(x, y, "r=", r[0], "rho=", rho[0])

             plt.subplot(1,2,yi+1)
             plt.scatter(results_df[x], results_df[y])
        
         plt.savefig(os.path.join(args.results_dir, x + '_corr.png'))
    '''
    # print(f'Results for model: {run_name}')

    if any(zero_mask):
        mean_decay = np.mean(results_df[zero_mask]['decay'])
        std_decay = np.std(results_df[zero_mask]['decay'])
        print(f'Decay: {mean_decay:.02f} +/- {std_decay:.02f}dB')
    
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
    
    return
    
    plt.clf()
    rooms = list(results_df['room'].unique())
    sisdri_list = []
    lines = []

    for room in rooms:
        df = results_df[results_df['room'] == room]
        mean_single_sisdri = np.mean(df['sisdri'])
        std_single_sisdri = np.std(df['sisdri'])

        # STOI
        mean_single_stoi = np.mean(df['stoi'])
        std_single_stoi = np.std(df['stoi'])
        print(f"[{room}] STOI: {mean_single_stoi:.03f}+/-{std_single_stoi}:.03f")

        # PESQ
        mean_single_pesq = np.mean(df['pesq'])
        std_single_pesq = np.std(df['pesq'])
        print(f"[{room}] PESQ: {mean_single_pesq:.03f}+/-{std_single_pesq}:.03f")

        print(room, mean_single_sisdri)
        specific_pos = {}

        angles = list(df["angle"])
        dises = list(df["dis"])
        sisdris = list(df['sisdri'])
        pesqs = list(df['pesq'])

        for i in range(len(angles)):
            a = angles[i]
            d = dises[i]
            pos_name = str(int(d))  + "_"+ str(int(a))
            if pos_name not in specific_pos.keys():
                temp = {}
                temp["SISDRi"] = [sisdris[i]]
                temp["PESQ"] = [pesqs[i]]
                specific_pos[pos_name] = temp
            else:
                specific_pos[pos_name]["SISDRi"].append(sisdris[i])
                specific_pos[pos_name]["PESQ"].append(pesqs[i])
        ''' 
        for name in specific_pos.keys():
            #print(name)
            #print("SISDRi = ", np.mean(specific_pos[name]["SISDRi"]))
            #print("PESQ = ", np.mean(specific_pos[name]["PESQ"]))
            lines.append(room + name + "; SISDRi = {:03f}".format(np.mean(specific_pos[name]["SISDRi"])) + "; PESQ = {:03f}".format(np.mean(specific_pos[name]["PESQ"]))  + "\n" )
        '''
        sisdri_list.append(mean_single_sisdri)

    #with open(args.results_dir + '/result.txt', 'w') as f:
    #    f.writelines(lines)

    plt.bar(rooms, sisdri_list)
    plt.xlabel('Room name')
    plt.ylabel('SI-SDRi')
    plt.savefig(os.path.join(args.results_dir, 'sisdri_vs_room.png'))
    plt.clf()

    for room in rooms:
        df = results_df[(results_df['room'] == room) & one_mask ]
        plt.scatter(df['input_sisdr'], df['sisdri'] + df['input_sisdr'], label=room, s=0.5)
    
    plt.plot([min(results_df[one_mask]['input_sisdr']), max(results_df[one_mask]['input_sisdr'])], 
             [min(results_df[one_mask]['input_sisdr']), max(results_df[one_mask]['input_sisdr'])], color='green')
    plt.legend()
    plt.xlabel('Input SI-SDR')
    plt.ylabel('Output SI-SDR')
    plt.savefig( os.path.join(args.results_dir, 'input_vs_output_si_sdr.png') )
    plt.clf()

    # print(results_df[results_df[one_mask]['sisdri'] < 15])
    room_mask = results_df['room'] == 'Daogao_cse674'
    print(results_df[room_mask])
    # print(results_df[room_mask & (results_df[room_mask]['input_sisdr'] < 2) ])
    # print(results_df[room_mask & (results_df[room_mask]['input_sisdr'] < 2) ].shape)
    # print(results_df[room_mask][results_df[room_mask]['tgt_speaker_ids'].apply(lambda x: 'p345' in x) ])

    tgt_speakers = list(results_df['tgt_speaker_ids'].unique())
    colors = []
    sisdris = []
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
