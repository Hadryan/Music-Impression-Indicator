from pydub import AudioSegment
from scipy.io import wavfile
import crepe
import os
import sys
import datetime
# import keyboard
import time
import threading
import shutil
import csv
import math

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

inputFile = 'input.wav'  # <<<<<   set new music file name

startTime = str(datetime.datetime.now())


RemoveWAVFolder = True
stopThreading = False
OS = ['linux', 'osx', 'win'][0]
printRefreshRate = 2.5  # 2.5 seconds
consoleOutputText = 'Started at: ' + str(startTime) + "\n\n\r"

samplesFiles = []
dirClassifications = './classifications'
dirTrainingData = './training-data'
dirConvertToWAV = dirTrainingData + '/to-wav/'

tracks = []  # [ [minimum-frequency , maximum-frequency] ]


ConfidenceRate = 0.68  # consider frequency values which has 0.68 confidence or more


def remove_wav_convert_folder():
    if os.path.exists(dirConvertToWAV) and RemoveWAVFolder == True:
        shutil.rmtree(dirConvertToWAV)


def printText():
    if stopThreading == False:
        threading.Timer(printRefreshRate, printText).start()

    if OS == 'win':
        os.system('cls')  # For Windows
    else:
        os.system('clear')  # For Linux/OS X

    sys.stdout.write("%s \r" % (consoleOutputText))
    sys.stdout.flush()


printText()
remove_wav_convert_folder()


# START Convert all media to WAV ------------------------------------


def convert_to_wav(fileName, dirDataFrom, dirConvertTo):
    file = os.path.join(dirDataFrom, fileName)
    fileNameWithoutExt = os.path.splitext(fileName)[0]
    fileExt = os.path.splitext(fileName)[1]

    if fileName.endswith('.wav'):
        return

    if fileExt.endswith('.mp3'):
        sound = AudioSegment.from_mp3(file)
    elif fileExt.endswith('.oog'):
        sound = AudioSegment.from_oog(file)
    else:
        try:
            sound = AudioSegment.from_file(file, fileExt)
        except:
            print(fileName + " is not format!")

    if not os.path.exists(dirConvertTo):
        place = os.mkdir(dirConvertTo, mode=0o777)
    convertedFile = dirConvertTo + fileNameWithoutExt + '.wav'
    sound.export(convertedFile, format="wav")


for fileName in os.listdir(dirTrainingData):
    if not fileName.endswith('.wav'):
        convert_to_wav(fileName, dirTrainingData, dirConvertToWAV)
        fileNameWithoutExt = os.path.splitext(fileName)[0]
        file = dirConvertToWAV + fileNameWithoutExt + '.wav'
        samplesFiles.append(file)
        continue
    samplesFiles.append(os.path.join(dirTrainingData, fileName))

# END Convert all media to WAV ------------------------------------


# print(samplesFiles)


consoleOutputText += 'Convert Files Done [100%]\n'


# START Extracting ------------------------------------


def cvs_handler(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        minFreq = math.inf
        maxFreq = -math.inf

        track = []

        lines = 0
        for row in csv_reader:
            if lines > 0:
                time, frequency, confidence = row
                if float(confidence) >= ConfidenceRate:
                    minFreq = min(minFreq, float(frequency))
                    maxFreq = max(maxFreq, float(frequency))
                    track.append([confidence, frequency])
            lines += 1

        return [minFreq, maxFreq]


def process_input(inputFileName):
    dirIn = os.path.join('./', '')
    convert_to_wav(inputFileName, dirIn, dirIn)
    file = os.path.join(dirIn, inputFileName)
    fileNameWithoutExt = os.path.splitext(inputFileName)[0]
    fileExt = os.path.splitext(inputFileName)[1]

    file.replace(fileExt, 'wav')
    inputFileName.replace(fileExt, 'wav')

    sr, audio = wavfile.read(file)
    time, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=True)
    consoleOutputText += '\nInput File: ' + inputFileName + ' Done [100%]\n'

    inputResult = cvs_handler(file)

    print(inputResult)


# Extract features from samples
for sample in samplesFiles:
    sr, audio = wavfile.read(sample)
    time, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=True)
    consoleOutputText += 'File: ' + sample + ' Done [100%]'


consoleOutputText += 'Predicting Process Finished!\n'


# Move csv file to classifications dir
currentDirFiles = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in currentDirFiles:
    if f.endswith('.csv'):
        shutil.move(f, dirClassifications)


consoleOutputText += 'csv files moved to path: ' + dirClassifications + "\n"


# Load data from csv files
for file in os.listdir(dirClassifications):
    file = os.path.join(dirClassifications, file)

    tracks.append(cvs_handler(file))

    # track.sort(reverse=True)
    # for r in track:
    #     print(r)

    # print(tracks)

consoleOutputText += 'Min and Max Frequencies Done!\n'


# Process Input
process_input(inputFile)


remove_wav_convert_folder()


# End
tillNowTime = str(datetime.timedelta(seconds=seconds))
stopThreading = True
consoleOutputText += 'Till now: ' + str(tillNowTime)
