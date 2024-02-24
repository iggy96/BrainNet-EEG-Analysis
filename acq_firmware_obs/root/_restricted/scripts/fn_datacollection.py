from fn_libs import *
from window_pptys import *

# used for testing without having to connect device
#def stimseq(eng_stim_dir, tone_duration, isi_tone, word_duration, isi_word, ibi, collection_duration):
#    print("eng_stim_dir:", eng_stim_dir)
#    return True  


def stimseq(txt_stim_dir, tone_duration, isi_tone, word_duration, isi_word, ibi, collection_duration):

    # Set the priority of the process to high
    current_process = psutil.Process(os.getpid())
    current_process.nice(psutil.REALTIME_PRIORITY_CLASS)

    def high_res_sleep(milliseconds):
        start_time = time.perf_counter()
        while True:
            current_time = time.perf_counter()
            elapsed_time = (current_time - start_time) * 1000  # convert to milliseconds
            if elapsed_time >= milliseconds:
                break

    def write_trigger_value(dev, trigger_value):

        # Set to the trigger_value
        trigger = bytes([trigger_value])
        dev.write(trigger)
 
        # keep the pin at the trigger state for 40ms (shaun's recommendation)
        high_res_sleep(80)

        # Set back to 0
        trigger = bytes([0])
        dev.write(trigger)

    def collect_eeg_data(d, samples, duration):
        #print("EEG data collection started")
        more = lambda s: samples.append(s.copy()) or len(samples) < duration
        data = d.GetData(d.SamplingRate, more)
        samples_data = np.concatenate(samples, axis=0)
        #print("EEG data collection completed")

    def play_audio_for_duration(audio_file, desired_duration):
        audio = AudioSegment.from_wav(audio_file)
        audio = audio.set_frame_rate(192000).set_channels(2)
        playback_obj = sa.play_buffer(audio.raw_data, num_channels=2, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
        time.sleep(desired_duration)
        playback_obj.stop()


    def play_block(dev, start_index, stimuli, trigger_vals, STIM_DIR, tone_duration, isi_tone, word_duration, isi_word, ibi):
        # Tones
        for i in range(4):
            #print(f"Playing stimulus: {stimuli[start_index + i]}")
            play_audio_for_duration(STIM_DIR + stimuli[start_index + i], tone_duration)  # Audio is played
            write_trigger_value(dev, trigger_vals[start_index + i])  # Trigger is sent after
            high_res_sleep(isi_tone * 1000)

        high_res_sleep(0.5 * 1000)

        # Words
        for i in range(2):
           # print(f"Playing stimulus: {stimuli[start_index + 4 + i]}")
            play_audio_for_duration(STIM_DIR + stimuli[start_index + 4 + i], word_duration)  # Audio is played
            write_trigger_value(dev, trigger_vals[start_index + 4 + i])  # Trigger is sent after
            high_res_sleep(isi_word * 1000)

        # Inter block interval
        high_res_sleep(ibi * 1000)
        
    def play_blocks(dev, stimuli, trigger_vals, STIM_DIR, tone_duration, isi_tone, word_duration, isi_word, ibi):
        # Loop through the stimuli array in blocks and play the audio while sending triggers
        for i in range(0, len(stimuli), 6):
            play_block(dev, i, stimuli, trigger_vals, STIM_DIR, tone_duration, isi_tone, word_duration, isi_word, ibi)


    def find_ftdi_device_by_description(target_description):
        """
        Find the index of the FTDI device with the given description.

        :param target_description: Description to search for.
        :return: index of the device or None if not found.
        """
        num_devices = FTD2XX.createDeviceInfoList()
        
        for index in range(num_devices):
            device_info = FTD2XX.getDeviceInfoDetail(index)
            if device_info['description'] == target_description:
                return index

        return None


    # File paths
    with open(r"..\assets\utils\biodata.txt", "r") as file:
        lines = [line.strip() for line in file.readlines()]
    stim_lang = lines[13].replace("Language:", "")

    if stim_lang == "English":
        TXT_FILE = txt_stim_dir
        STIM_DIR = r"..\assets\sounds\english\\"

    elif stim_lang == "Chichenwa":
        TXT_FILE = txt_stim_dir
        STIM_DIR = r"..\assets\sounds\chichenwa\\"

    # Read the text file
    stimuli, trigger_vals = [], []
    with open(TXT_FILE, 'r') as file:
        for line in file:
            elements = line.strip().split(', ')
            # if the len(elements) is 0 or 1, the use split(',')
            if len(elements) == 0 or len(elements) == 1:
                elements = line.strip().split(',')
            else:
                pass

            for i, element in enumerate(elements):
                if i % 2 == 0:
                    stimuli.append(element + ".wav")
                else:
                    trigger_vals.append(int(element))

    # Trigger device configuration
#    ftd = FTD2XX.open(1)  # open the first device but the trigger device is the second device
    target_description = b'TTL232RG-VSW3V3'
    device_index = find_ftdi_device_by_description(target_description)
    ftd = FTD2XX.open(device_index)
    high_res_sleep(50)
    ftd.setBitMode(0xFF, 1)
    high_res_sleep(50)

    # EEG device configuration
    d = g.GDS()
    minf_s = sorted(d.GetSupportedSamplingRates()[0].items())[1]
    d.SamplingRate, d.NumberOfScans = minf_s
    d.Trigger = 1
    for ch in d.Channels:
        ch.Acquire = True
        ch.BipolarChannel = -1
        ch.BandpassFilterIndex = -1
        ch.NotchFilterIndex = -1
    d.SetConfiguration()

    samples = []


    # convert string to float
    tone_duration = float(tone_duration)
    isi_tone = float(isi_tone)
    word_duration = float(word_duration)
    isi_word = float(isi_word)
    ibi = float(ibi)
    collection_duration = int(collection_duration)

    # Start the EEG data collection thread
    eeg_thread = threading.Thread(target=collect_eeg_data, args=(d, samples, collection_duration))
    eeg_thread.start()
   # print("EEG data collection thread started")

    # Start the play blocks thread
    high_res_sleep(1000) # this allows the eeg start collecting data 1 second before the audio starts playing
    play_blocks_thread = threading.Thread(target=play_blocks, args=(ftd, stimuli, trigger_vals, STIM_DIR, tone_duration, isi_tone, word_duration, isi_word, ibi))
    play_blocks_thread.start()
   # print("Play blocks thread started")

    # Wait for both threads to complete
    eeg_thread.join()
    play_blocks_thread.join()
   # print("EEG data collection and play blocks threads completed")

    # Close the FTDI device
    ftd.close()

    # Concatenate the samples data
    samples_data = np.concatenate(samples, axis=0)

    # Read biodata.txt file
    with open(r"..\assets\utils\biodata.txt", "r") as file:
        lines = [line.strip() for line in file.readlines()]
    sid = lines[1].replace("ID:", "")
    sdate = lines[4].replace("Date:", "")
    path = r"..\..\user_access\raw"

    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    filename = f'{sid}_{sdate}_1.npy'

    counter = 1
    base_name, ext = os.path.splitext(filename)
    while os.path.exists(os.path.join(path, filename)):
        counter += 1
        filename = f"{base_name[:-1]}{counter}{ext}"


    full_path = os.path.join(path, filename)

    # Assuming `samples_data` is the data you want to save
    np.save(full_path, samples_data)

    d.Close()
    del d





