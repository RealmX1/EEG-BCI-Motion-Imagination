This README.txt file was generated on 20250529 by Yidan Ding

#
# General instructions for completing README: 
# For sections that are non-applicable, mark as N/A (do not delete any sections). 
# Please leave all commented sections in README (do not delete any text). 
#

-------------------
GENERAL INFORMATION
-------------------

1. Title of Dataset: EEG-BCI Dataset for Real-time Robotic Hand Control at Individual Finger Level

#
# Authors: Include contact information for at least the 
# first author and corresponding author (if not the same), 
# specifically email address, phone number (optional, but preferred), and institution. 
# Contact information for all authors is preferred.
#

2. Author Information
<create a new entry for each additional author>

First Author Contact Information
    Name: Yidan Ding
    ORCiD: 0009-0003-9363-7755
    Institution: Department of Biomedical Engineering, Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, PA 15213, USA
    Email: yidand@andrew.cmu.edu
    Phone Number: 


Corresponding Author Contact Information
    Name: Bin He
    ORCiD: 0000-0003-2944-8602
    Institution: Department of Biomedical Engineering, Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, PA 15213, USA
    Email: bhe1@andrew.cmu.edu
    Phone Number: 412 268 9857

Author Contact Information (if applicable)
    Name:
    ORCiD:
    Institution:
    Address:
    Email:
    Phone Number: 

---------------------
DATA & FILE OVERVIEW
---------------------

#
# Directory of Files in Dataset: List and define the different 
# files included in the dataset. This serves as its table of 
# contents. 
#

Directory of Files:
   A. Filename: SXX/TaskType(_SessYY_Zclass_Model)/SXX_TaskType(_SessYY_Zclass_Model)_RXX.mat        
      Short description: MATLAB object files containing EEG data and event information for each experimental run        
        
   B. Filename: biosemi128.ELC        
      Short description: Electrode configuration file for the 128-channel BioSemi EEG headcap        


Additional Notes on File Relationships, Context, or Content 
(for example, if a user wants to reuse and/or cite your data, 
what information would you want them to know?):              
This dataset contains EEG recordings of subjects controlling a robotic hand at the individual finger level using motor execution (ME) or motor imagery (MI). The data is organized by subject, session type, and experimental run. Each .mat file contains continuous EEG data from a single experimental run along with associated event markers and (for online sessions) prediction information.


#
# File Naming Convention: Define your File Naming Convention 
# (FNC), the framework used for naming your files systematically 
# to describe what they contain, which could be combined with the
# Directory of Files. 
#

File Naming Convention:
- Main folders are named as SXX where XX refers to the subject ID ranging from 01 to 21
- Subfolders are named as TaskType(_SessYY_Zclass_Model). Each subfolder contains data from the same task condition within the same session.
  - TaskType indicates whether it's online or offline, movement or imagery, and with or without the smoothing mechanism (e.g., OfflineImagery, OnlineSmoothMovement)
  - SessYY indicates session number (YY ranges from 01 up to 05) if multiple sessions are involved
  - Zclass distinguishes between 2-class and 3-class control paradigms for online sessions (Z is either 2 or 3)
  - Model specifies decoding approach (Base, Finetune, or Smooth)
- Individual files are named with run number indicated by suffix RXX


#
# Data Description: A data description, dictionary, or codebook
# defines the variables and abbreviations used in a dataset. This
# information can be included in the README file, in a separate 
# file, or as part of the data file. If it is in a separate file
# or in the data file, explain where this information is located
# and ensure that it is accessible without specialized software.
# (We recommend using plain text files or tabular plain text CSV
# files exported from spreadsheet software.) 
#

-----------------------------------------
DATA DESCRIPTION FOR: SXX_TaskType(_SessYY_Zclass_Model)_RXX.mat
-----------------------------------------

1. Number of variables: 2 main MATLAB structures (eeg and event)


2. Number of cases/rows: Varies by run


3. Missing data codes:
        N/A        N/A
        Note: Event information for S07\OnlineImagery_Sess05_3class_Base is missing, making data from these runs potentially unusable.


4. Variable List

#
# Example. Name: Wall Type 
#     Description: The type of materials present in the wall type for housing surveys collected in the project.
#         1 = Brick
#         2 = Concrete blocks
#	  3 = Clay
#         4 = Steel panels


    A. Name: eeg
       Description: MATLAB structure containing the following fields:
                    data - (matrix of double) Matrix (128 channels × time points) containing raw continuous EEG values
                    time - (vector of double) Vector of timestamps (in seconds) corresponding to EEG data points
                    label - (vector of cells) Vector of cells containing channel labels
                    fsample - (double) Sampling frequency in Hz (1,024 Hz)
                    nChans - (double) Number of channels (128)
                    nSamples - (double) Number of time points
                    prediction - (vector of int) (Online sessions only) Vector of predicted finger class at each time point
                                 1 = Thumb, 2 = Index Finger, 4 = Pinky
                    prob_thumb - (vector of int) (Online sessions only) Predicted probability for Thumb class
                    prob_index - (vector of int) (Online sessions, 3-class only) Predicted probability for Index Finger class
                    prob_pinky - (vector of int) (Online sessions only) Predicted probability for Pinky class
		    Note: The online scripts process the data pockets every 125 ms, so the predicted class and probabilities update every 128 samples. Given that the continuous data contains both the data during the trials and the inter-trial intervals, the values during the inter-trial intervals might not be meaningful.


    B. Name: event
       Description: MATLAB structure array containing the following fields:
                    type - Event type ("Target" for trial start, "TrialEnd" for trial end)
                    sample - Timestamp of the event
                    value - For "Target" events, indicates target class (1 = Thumb, 2 = Index Finger, 4 = Pinky)

--------------------------
METHODOLOGICAL INFORMATION
--------------------------

#
# Software: If specialized software(s) generated your data or
# are necessary to interpret it, please provide for each (if
# applicable): software name, version, system requirements,
# and developer. 
#If you developed the software, please provide (if applicable): 
#A copy of the software's binary executable compatible with the system requirements described above. 
#A source snapshot or distribution if the source code is not stored in a publicly available online repository.
#All software source components, including pointers to source(s) for third-party components (if any)

1. Software-specific information:
<create a new entry for each qualifying software program>

Name: MATLAB
Version: R2023a
System Requirements: N
Open Source? (Y/N): N

(if available and applicable)
Executable URL:
Source Repository URL:
Developer: MathWorks
Product URL:
Software source components:


Additional Notes(such as, will this software not run on 
certain operating systems?):


#
# Equipment: If specialized equipment generated your data,
# please provide for each (if applicable): equipment name,
# manufacturer, model, and calibration information. Be sure
# to include specialized file format information in the data
# dictionary.
#

2. Equipment-specific information:
<create a new entry for each qualifying piece of equipment>

Manufacturer: BioSemi, Amsterdam, The Netherlands
Model: ActiveTwo amplifier with 128-channel EEG headcap

(if applicable)
Embedded Software / Firmware Name: ActiView
Embedded Software / Firmware Version: 9.02 (Win64)
Additional Notes: EEG signals were recorded at a sampling rate of 1,024 Hz

#
# Dates of Data Collection: List the dates and/or times of
# data collection.
#

3. Date of data collection (single date, range, approximate date) <suggested format YYYYMMDD>: 20231126 - 20250403

--------------------------
EXPERIMENTAL DESIGN
--------------------------

Study Overview:
Twenty-one right-handed subjects participated in experiments controlling an EEG-based BCI using motor execution (ME) or motor imagery (MI) of their fingers within their dominant hand (right hand) to control corresponding finger motions of a robotic hand in real time.

Session Types:

1. Offline Sessions:
   - Subjects instructed to execute or imagine repetitive single-finger flexion and extension
   - Fingers included: thumb, index finger, middle finger, and pinky
   - 32 runs per session, 5 trials per finger in randomized order (or 30 for subjects who requested practice runs)
   - Each trial lasted 5 seconds, followed by 2-second inter-trial interval

2. Online Sessions:
   - Similar finger movement/imagery tasks as offline sessions
   - Two paradigms: binary classification (thumb vs. pinky) and ternary classification (thumb, index, pinky)
   - 32 runs per session 
   - Run distribution: 8 runs ternary with base model, 8 runs binary with base model, 8 runs ternary with fine-tuned model, 8 runs binary with fine-tuned model
   - Each trial lasted 3 seconds with 1-second delay before 2-second feedback period
   - 2-second inter-trial interval between consecutive trials

3. Online Sessions with Smoothed Robotic Control:
   - Similar to regular online sessions
   - Last 8 runs using fine-tuned decoder were split: 4 runs with original algorithm, 4 runs with smoothing mechanism

Participation Details:
- All subjects completed one offline and two online sessions each for ME and MI tasks
- Sixteen subjects completed three additional MI online sessions
- These sixteen subjects also completed two more online sessions with smoothed robotic control (one ME, one MI)
