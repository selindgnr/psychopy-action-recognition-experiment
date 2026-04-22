#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Mayis 14, 2025, at 14:48
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'action_recognition_experiment'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='action_recognition_experiment.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_instruction') is None:
        # initialise key_resp_instruction
        key_resp_instruction = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_instruction',
        )
    if deviceManager.getDevice('key_resp_instruction_2') is None:
        # initialise key_resp_instruction_2
        key_resp_instruction_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_instruction_2',
        )
    if deviceManager.getDevice('testResponse') is None:
        # initialise testResponse
        testResponse = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='testResponse',
        )
    if deviceManager.getDevice('breakKey') is None:
        # initialise breakKey
        breakKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='breakKey',
        )
    if deviceManager.getDevice('key_resp_instruction_3') is None:
        # initialise key_resp_instruction_3
        key_resp_instruction_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_instruction_3',
        )
    if deviceManager.getDevice('testResponseAFD') is None:
        # initialise testResponseAFD
        testResponseAFD = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='testResponseAFD',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructionsRoutine" ---
    instruction_text = visual.TextStim(win=win, name='instruction_text',
        text='Welcome to the experiment!\n\nYou will see a series of short videos showing human actions.\n\nYour task is to identify the action shown in each video.\nEach action is mapped to a number key (1–5):\n\n1 = JumpingJack  \n2 = Lunges  \n3 = PullUps  \n4 = PushUps  \n5 = Swing\n\nUse the number keys (1 to 5) on your keyboard to respond.\n\nPress SPACE to begin the training phase.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instruction = keyboard.Keyboard(deviceName='key_resp_instruction')
    
    # --- Initialize components for Routine "ucf5TrainRoutine" ---
    trainVideo = visual.MovieStim(
        win, name='trainVideo',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=(0.5, 0.5), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    trainLabel = visual.TextStim(win=win, name='trainLabel',
        text='',
        font='Arial',
        pos=[0, -0.4], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "ucfinstructions" ---
    instruction_text_2 = visual.TextStim(win=win, name='instruction_text_2',
        text='You will see a series of short videos showing human actions.\n\nYour task is to identify the action shown in each video.\nEach action is mapped to a number key (1–5):\n\n1 = JumpingJack  \n2 = Lunges  \n3 = PullUps  \n4 = PushUps  \n5 = Swing\n\nUse the number keys (1 to 5) on your keyboard to respond.\n\nPress SPACE to begin the training phase.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instruction_2 = keyboard.Keyboard(deviceName='key_resp_instruction_2')
    
    # --- Initialize components for Routine "ucf5TestRoutine" ---
    testVideo = visual.MovieStim(
        win, name='testVideo',
        filename=None, movieLib='ffpyplayer',
        loop=True, volume=1.0, noAudio=False,
        pos=(0, 0), size=(0.5, 0.5), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    promptText = visual.TextStim(win=win, name='promptText',
        text='Classify the action:\n1 = JumpingJack\n2 = Lunges\n3 = PullUps\n4 = PushUps\n5 = Swing',
        font='Arial',
        pos=[0, -0.4], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    testResponse = keyboard.Keyboard(deviceName='testResponse')
    
    # --- Initialize components for Routine "breakRoutine" ---
    breakText = visual.TextStim(win=win, name='breakText',
        text='You can take a short break now.\nWhen you are ready, press SPACE to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    breakKey = keyboard.Keyboard(deviceName='breakKey')
    
    # --- Initialize components for Routine "afd5TrainRoutine" ---
    trainVideoAFD = visual.MovieStim(
        win, name='trainVideoAFD',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=(0.5, 0.5), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    trainLabelAFD = visual.TextStim(win=win, name='trainLabelAFD',
        text='',
        font='Arial',
        pos=[0, -0.4], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "afdinstructions" ---
    key_resp_instruction_3 = keyboard.Keyboard(deviceName='key_resp_instruction_3')
    instruction_text_3 = visual.TextStim(win=win, name='instruction_text_3',
        text='You will see a series of short videos showing human actions.\n\nYour task is to identify the action shown in each video.\nEach action is mapped to a number key (1–5):\n\n1 = JumpingJack  \n2 = Lunges  \n3 = PullUps  \n4 = PushUps  \n5 = Swing\n\nUse the number keys (1 to 5) on your keyboard to respond.\n\nPress SPACE to begin the training phase.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "afd5TestRoutine" ---
    testVideoAFD = visual.MovieStim(
        win, name='testVideoAFD',
        filename=None, movieLib='ffpyplayer',
        loop=True, volume=1.0, noAudio=False,
        pos=(0, 0), size=(0.5, 0.5), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    promptAFD = visual.TextStim(win=win, name='promptAFD',
        text='Classify the action:\n1 = JumpingJack\n2 = Lunges\n3 = PullUps\n4 = PushUps\n5 = Swing\n',
        font='Arial',
        pos=[0, -0.4], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    testResponseAFD = keyboard.Keyboard(deviceName='testResponseAFD')
    
    # --- Initialize components for Routine "goodbyeRoutine" ---
    goodbyetext = visual.TextStim(win=win, name='goodbyetext',
        text='Thank you for participating!  \nYou may now close this window by pressing SPACE.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructionsRoutine" ---
    # create an object to store info about Routine instructionsRoutine
    instructionsRoutine = data.Routine(
        name='instructionsRoutine',
        components=[instruction_text, key_resp_instruction],
    )
    instructionsRoutine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instruction
    key_resp_instruction.keys = []
    key_resp_instruction.rt = []
    _key_resp_instruction_allKeys = []
    # store start times for instructionsRoutine
    instructionsRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructionsRoutine.tStart = globalClock.getTime(format='float')
    instructionsRoutine.status = STARTED
    thisExp.addData('instructionsRoutine.started', instructionsRoutine.tStart)
    instructionsRoutine.maxDuration = None
    # keep track of which components have finished
    instructionsRoutineComponents = instructionsRoutine.components
    for thisComponent in instructionsRoutine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructionsRoutine" ---
    instructionsRoutine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_text* updates
        
        # if instruction_text is starting this frame...
        if instruction_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_text.frameNStart = frameN  # exact frame index
            instruction_text.tStart = t  # local t and not account for scr refresh
            instruction_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_text.started')
            # update status
            instruction_text.status = STARTED
            instruction_text.setAutoDraw(True)
        
        # if instruction_text is active this frame...
        if instruction_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_instruction* updates
        waitOnFlip = False
        
        # if key_resp_instruction is starting this frame...
        if key_resp_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instruction.frameNStart = frameN  # exact frame index
            key_resp_instruction.tStart = t  # local t and not account for scr refresh
            key_resp_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_instruction.started')
            # update status
            key_resp_instruction.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_instruction.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_instruction.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_instruction.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_instruction.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instruction_allKeys.extend(theseKeys)
            if len(_key_resp_instruction_allKeys):
                key_resp_instruction.keys = _key_resp_instruction_allKeys[-1].name  # just the last key pressed
                key_resp_instruction.rt = _key_resp_instruction_allKeys[-1].rt
                key_resp_instruction.duration = _key_resp_instruction_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructionsRoutine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsRoutine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructionsRoutine" ---
    for thisComponent in instructionsRoutine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructionsRoutine
    instructionsRoutine.tStop = globalClock.getTime(format='float')
    instructionsRoutine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructionsRoutine.stopped', instructionsRoutine.tStop)
    # check responses
    if key_resp_instruction.keys in ['', [], None]:  # No response was made
        key_resp_instruction.keys = None
    thisExp.addData('key_resp_instruction.keys',key_resp_instruction.keys)
    if key_resp_instruction.keys != None:  # we had a response
        thisExp.addData('key_resp_instruction.rt', key_resp_instruction.rt)
        thisExp.addData('key_resp_instruction.duration', key_resp_instruction.duration)
    thisExp.nextEntry()
    # the Routine "instructionsRoutine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    ucf5TrainLoop = data.TrialHandler2(
        name='ucf5TrainLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('ucf5_training_conditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(ucf5TrainLoop)  # add the loop to the experiment
    thisUcf5TrainLoop = ucf5TrainLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisUcf5TrainLoop.rgb)
    if thisUcf5TrainLoop != None:
        for paramName in thisUcf5TrainLoop:
            globals()[paramName] = thisUcf5TrainLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisUcf5TrainLoop in ucf5TrainLoop:
        currentLoop = ucf5TrainLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisUcf5TrainLoop.rgb)
        if thisUcf5TrainLoop != None:
            for paramName in thisUcf5TrainLoop:
                globals()[paramName] = thisUcf5TrainLoop[paramName]
        
        # --- Prepare to start Routine "ucf5TrainRoutine" ---
        # create an object to store info about Routine ucf5TrainRoutine
        ucf5TrainRoutine = data.Routine(
            name='ucf5TrainRoutine',
            components=[trainVideo, trainLabel],
        )
        ucf5TrainRoutine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        trainVideo.setMovie(video_file)
        trainLabel.setText(label)
        # store start times for ucf5TrainRoutine
        ucf5TrainRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ucf5TrainRoutine.tStart = globalClock.getTime(format='float')
        ucf5TrainRoutine.status = STARTED
        thisExp.addData('ucf5TrainRoutine.started', ucf5TrainRoutine.tStart)
        ucf5TrainRoutine.maxDuration = None
        # keep track of which components have finished
        ucf5TrainRoutineComponents = ucf5TrainRoutine.components
        for thisComponent in ucf5TrainRoutine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ucf5TrainRoutine" ---
        # if trial has changed, end Routine now
        if isinstance(ucf5TrainLoop, data.TrialHandler2) and thisUcf5TrainLoop.thisN != ucf5TrainLoop.thisTrial.thisN:
            continueRoutine = False
        ucf5TrainRoutine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *trainVideo* updates
            
            # if trainVideo is starting this frame...
            if trainVideo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trainVideo.frameNStart = frameN  # exact frame index
                trainVideo.tStart = t  # local t and not account for scr refresh
                trainVideo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trainVideo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trainVideo.started')
                # update status
                trainVideo.status = STARTED
                trainVideo.setAutoDraw(True)
                trainVideo.play()
            
            # if trainVideo is stopping this frame...
            if trainVideo.status == STARTED:
                if bool(False) or trainVideo.isFinished:
                    # keep track of stop time/frame for later
                    trainVideo.tStop = t  # not accounting for scr refresh
                    trainVideo.tStopRefresh = tThisFlipGlobal  # on global time
                    trainVideo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trainVideo.stopped')
                    # update status
                    trainVideo.status = FINISHED
                    trainVideo.setAutoDraw(False)
                    trainVideo.stop()
            if trainVideo.isFinished:  # force-end the Routine
                continueRoutine = False
            
            # *trainLabel* updates
            
            # if trainLabel is starting this frame...
            if trainLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trainLabel.frameNStart = frameN  # exact frame index
                trainLabel.tStart = t  # local t and not account for scr refresh
                trainLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trainLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trainLabel.started')
                # update status
                trainLabel.status = STARTED
                trainLabel.setAutoDraw(True)
            
            # if trainLabel is active this frame...
            if trainLabel.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[trainVideo]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                ucf5TrainRoutine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ucf5TrainRoutine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ucf5TrainRoutine" ---
        for thisComponent in ucf5TrainRoutine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ucf5TrainRoutine
        ucf5TrainRoutine.tStop = globalClock.getTime(format='float')
        ucf5TrainRoutine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('ucf5TrainRoutine.stopped', ucf5TrainRoutine.tStop)
        trainVideo.stop()  # ensure movie has stopped at end of Routine
        # the Routine "ucf5TrainRoutine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'ucf5TrainLoop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "ucfinstructions" ---
    # create an object to store info about Routine ucfinstructions
    ucfinstructions = data.Routine(
        name='ucfinstructions',
        components=[instruction_text_2, key_resp_instruction_2],
    )
    ucfinstructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instruction_2
    key_resp_instruction_2.keys = []
    key_resp_instruction_2.rt = []
    _key_resp_instruction_2_allKeys = []
    # store start times for ucfinstructions
    ucfinstructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ucfinstructions.tStart = globalClock.getTime(format='float')
    ucfinstructions.status = STARTED
    thisExp.addData('ucfinstructions.started', ucfinstructions.tStart)
    ucfinstructions.maxDuration = None
    # keep track of which components have finished
    ucfinstructionsComponents = ucfinstructions.components
    for thisComponent in ucfinstructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ucfinstructions" ---
    ucfinstructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_text_2* updates
        
        # if instruction_text_2 is starting this frame...
        if instruction_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_text_2.frameNStart = frameN  # exact frame index
            instruction_text_2.tStart = t  # local t and not account for scr refresh
            instruction_text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_text_2.started')
            # update status
            instruction_text_2.status = STARTED
            instruction_text_2.setAutoDraw(True)
        
        # if instruction_text_2 is active this frame...
        if instruction_text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_instruction_2* updates
        waitOnFlip = False
        
        # if key_resp_instruction_2 is starting this frame...
        if key_resp_instruction_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instruction_2.frameNStart = frameN  # exact frame index
            key_resp_instruction_2.tStart = t  # local t and not account for scr refresh
            key_resp_instruction_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instruction_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_instruction_2.started')
            # update status
            key_resp_instruction_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_instruction_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_instruction_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_instruction_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_instruction_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instruction_2_allKeys.extend(theseKeys)
            if len(_key_resp_instruction_2_allKeys):
                key_resp_instruction_2.keys = _key_resp_instruction_2_allKeys[-1].name  # just the last key pressed
                key_resp_instruction_2.rt = _key_resp_instruction_2_allKeys[-1].rt
                key_resp_instruction_2.duration = _key_resp_instruction_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            ucfinstructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ucfinstructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ucfinstructions" ---
    for thisComponent in ucfinstructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ucfinstructions
    ucfinstructions.tStop = globalClock.getTime(format='float')
    ucfinstructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ucfinstructions.stopped', ucfinstructions.tStop)
    # check responses
    if key_resp_instruction_2.keys in ['', [], None]:  # No response was made
        key_resp_instruction_2.keys = None
    thisExp.addData('key_resp_instruction_2.keys',key_resp_instruction_2.keys)
    if key_resp_instruction_2.keys != None:  # we had a response
        thisExp.addData('key_resp_instruction_2.rt', key_resp_instruction_2.rt)
        thisExp.addData('key_resp_instruction_2.duration', key_resp_instruction_2.duration)
    thisExp.nextEntry()
    # the Routine "ucfinstructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    ucf5TestLoop = data.TrialHandler2(
        name='ucf5TestLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('sorted_test_list.csv'), 
        seed=None, 
    )
    thisExp.addLoop(ucf5TestLoop)  # add the loop to the experiment
    thisUcf5TestLoop = ucf5TestLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisUcf5TestLoop.rgb)
    if thisUcf5TestLoop != None:
        for paramName in thisUcf5TestLoop:
            globals()[paramName] = thisUcf5TestLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisUcf5TestLoop in ucf5TestLoop:
        currentLoop = ucf5TestLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisUcf5TestLoop.rgb)
        if thisUcf5TestLoop != None:
            for paramName in thisUcf5TestLoop:
                globals()[paramName] = thisUcf5TestLoop[paramName]
        
        # --- Prepare to start Routine "ucf5TestRoutine" ---
        # create an object to store info about Routine ucf5TestRoutine
        ucf5TestRoutine = data.Routine(
            name='ucf5TestRoutine',
            components=[testVideo, promptText, testResponse],
        )
        ucf5TestRoutine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        testVideo.setMovie(video_file)
        # create starting attributes for testResponse
        testResponse.keys = []
        testResponse.rt = []
        _testResponse_allKeys = []
        # store start times for ucf5TestRoutine
        ucf5TestRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ucf5TestRoutine.tStart = globalClock.getTime(format='float')
        ucf5TestRoutine.status = STARTED
        thisExp.addData('ucf5TestRoutine.started', ucf5TestRoutine.tStart)
        ucf5TestRoutine.maxDuration = None
        # keep track of which components have finished
        ucf5TestRoutineComponents = ucf5TestRoutine.components
        for thisComponent in ucf5TestRoutine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ucf5TestRoutine" ---
        # if trial has changed, end Routine now
        if isinstance(ucf5TestLoop, data.TrialHandler2) and thisUcf5TestLoop.thisN != ucf5TestLoop.thisTrial.thisN:
            continueRoutine = False
        ucf5TestRoutine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *testVideo* updates
            
            # if testVideo is starting this frame...
            if testVideo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                testVideo.frameNStart = frameN  # exact frame index
                testVideo.tStart = t  # local t and not account for scr refresh
                testVideo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(testVideo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'testVideo.started')
                # update status
                testVideo.status = STARTED
                testVideo.setAutoDraw(True)
                testVideo.play()
            
            # if testVideo is stopping this frame...
            if testVideo.status == STARTED:
                if bool(False) or testVideo.isFinished:
                    # keep track of stop time/frame for later
                    testVideo.tStop = t  # not accounting for scr refresh
                    testVideo.tStopRefresh = tThisFlipGlobal  # on global time
                    testVideo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'testVideo.stopped')
                    # update status
                    testVideo.status = FINISHED
                    testVideo.setAutoDraw(False)
                    testVideo.stop()
            if testVideo.isFinished:  # force-end the Routine
                continueRoutine = False
            
            # *promptText* updates
            
            # if promptText is starting this frame...
            if promptText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                promptText.frameNStart = frameN  # exact frame index
                promptText.tStart = t  # local t and not account for scr refresh
                promptText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(promptText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'promptText.started')
                # update status
                promptText.status = STARTED
                promptText.setAutoDraw(True)
            
            # if promptText is active this frame...
            if promptText.status == STARTED:
                # update params
                pass
            
            # *testResponse* updates
            waitOnFlip = False
            
            # if testResponse is starting this frame...
            if testResponse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                testResponse.frameNStart = frameN  # exact frame index
                testResponse.tStart = t  # local t and not account for scr refresh
                testResponse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(testResponse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'testResponse.started')
                # update status
                testResponse.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(testResponse.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(testResponse.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if testResponse.status == STARTED and not waitOnFlip:
                theseKeys = testResponse.getKeys(keyList=['1', '2', '3', '4', '5'], ignoreKeys=["escape"], waitRelease=False)
                _testResponse_allKeys.extend(theseKeys)
                if len(_testResponse_allKeys):
                    testResponse.keys = _testResponse_allKeys[-1].name  # just the last key pressed
                    testResponse.rt = _testResponse_allKeys[-1].rt
                    testResponse.duration = _testResponse_allKeys[-1].duration
                    # was this correct?
                    if (testResponse.keys == str(correct_ans)) or (testResponse.keys == correct_ans):
                        testResponse.corr = 1
                    else:
                        testResponse.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[testVideo]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                ucf5TestRoutine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ucf5TestRoutine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ucf5TestRoutine" ---
        for thisComponent in ucf5TestRoutine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ucf5TestRoutine
        ucf5TestRoutine.tStop = globalClock.getTime(format='float')
        ucf5TestRoutine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('ucf5TestRoutine.stopped', ucf5TestRoutine.tStop)
        testVideo.stop()  # ensure movie has stopped at end of Routine
        # check responses
        if testResponse.keys in ['', [], None]:  # No response was made
            testResponse.keys = None
            # was no response the correct answer?!
            if str(correct_ans).lower() == 'none':
               testResponse.corr = 1;  # correct non-response
            else:
               testResponse.corr = 0;  # failed to respond (incorrectly)
        # store data for ucf5TestLoop (TrialHandler)
        ucf5TestLoop.addData('testResponse.keys',testResponse.keys)
        ucf5TestLoop.addData('testResponse.corr', testResponse.corr)
        if testResponse.keys != None:  # we had a response
            ucf5TestLoop.addData('testResponse.rt', testResponse.rt)
            ucf5TestLoop.addData('testResponse.duration', testResponse.duration)
        # Run 'End Routine' code from testEval
        # Compare participant response to correct key
        if testResponse.keys == str(correct_ans):
            correct = 1
        else:
            correct = 0
        
        # Save to the output file
        thisExp.addData('correct', correct)
        thisExp.addData('given_answer', testResponse.keys)
        # the Routine "ucf5TestRoutine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'ucf5TestLoop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "breakRoutine" ---
    # create an object to store info about Routine breakRoutine
    breakRoutine = data.Routine(
        name='breakRoutine',
        components=[breakText, breakKey],
    )
    breakRoutine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for breakKey
    breakKey.keys = []
    breakKey.rt = []
    _breakKey_allKeys = []
    # store start times for breakRoutine
    breakRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    breakRoutine.tStart = globalClock.getTime(format='float')
    breakRoutine.status = STARTED
    thisExp.addData('breakRoutine.started', breakRoutine.tStart)
    breakRoutine.maxDuration = None
    # keep track of which components have finished
    breakRoutineComponents = breakRoutine.components
    for thisComponent in breakRoutine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "breakRoutine" ---
    breakRoutine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *breakText* updates
        
        # if breakText is starting this frame...
        if breakText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            breakText.frameNStart = frameN  # exact frame index
            breakText.tStart = t  # local t and not account for scr refresh
            breakText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(breakText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'breakText.started')
            # update status
            breakText.status = STARTED
            breakText.setAutoDraw(True)
        
        # if breakText is active this frame...
        if breakText.status == STARTED:
            # update params
            pass
        
        # *breakKey* updates
        waitOnFlip = False
        
        # if breakKey is starting this frame...
        if breakKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            breakKey.frameNStart = frameN  # exact frame index
            breakKey.tStart = t  # local t and not account for scr refresh
            breakKey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(breakKey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'breakKey.started')
            # update status
            breakKey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(breakKey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(breakKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if breakKey.status == STARTED and not waitOnFlip:
            theseKeys = breakKey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _breakKey_allKeys.extend(theseKeys)
            if len(_breakKey_allKeys):
                breakKey.keys = _breakKey_allKeys[-1].name  # just the last key pressed
                breakKey.rt = _breakKey_allKeys[-1].rt
                breakKey.duration = _breakKey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            breakRoutine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in breakRoutine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "breakRoutine" ---
    for thisComponent in breakRoutine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for breakRoutine
    breakRoutine.tStop = globalClock.getTime(format='float')
    breakRoutine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('breakRoutine.stopped', breakRoutine.tStop)
    # check responses
    if breakKey.keys in ['', [], None]:  # No response was made
        breakKey.keys = None
    thisExp.addData('breakKey.keys',breakKey.keys)
    if breakKey.keys != None:  # we had a response
        thisExp.addData('breakKey.rt', breakKey.rt)
        thisExp.addData('breakKey.duration', breakKey.duration)
    thisExp.nextEntry()
    # the Routine "breakRoutine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    afd5TrainLoop = data.TrialHandler2(
        name='afd5TrainLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('afd5_training_conditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(afd5TrainLoop)  # add the loop to the experiment
    thisAfd5TrainLoop = afd5TrainLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisAfd5TrainLoop.rgb)
    if thisAfd5TrainLoop != None:
        for paramName in thisAfd5TrainLoop:
            globals()[paramName] = thisAfd5TrainLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisAfd5TrainLoop in afd5TrainLoop:
        currentLoop = afd5TrainLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisAfd5TrainLoop.rgb)
        if thisAfd5TrainLoop != None:
            for paramName in thisAfd5TrainLoop:
                globals()[paramName] = thisAfd5TrainLoop[paramName]
        
        # --- Prepare to start Routine "afd5TrainRoutine" ---
        # create an object to store info about Routine afd5TrainRoutine
        afd5TrainRoutine = data.Routine(
            name='afd5TrainRoutine',
            components=[trainVideoAFD, trainLabelAFD],
        )
        afd5TrainRoutine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        trainVideoAFD.setMovie(video_file)
        trainLabelAFD.setText(label)
        # store start times for afd5TrainRoutine
        afd5TrainRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        afd5TrainRoutine.tStart = globalClock.getTime(format='float')
        afd5TrainRoutine.status = STARTED
        thisExp.addData('afd5TrainRoutine.started', afd5TrainRoutine.tStart)
        afd5TrainRoutine.maxDuration = None
        # keep track of which components have finished
        afd5TrainRoutineComponents = afd5TrainRoutine.components
        for thisComponent in afd5TrainRoutine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "afd5TrainRoutine" ---
        # if trial has changed, end Routine now
        if isinstance(afd5TrainLoop, data.TrialHandler2) and thisAfd5TrainLoop.thisN != afd5TrainLoop.thisTrial.thisN:
            continueRoutine = False
        afd5TrainRoutine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *trainVideoAFD* updates
            
            # if trainVideoAFD is starting this frame...
            if trainVideoAFD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trainVideoAFD.frameNStart = frameN  # exact frame index
                trainVideoAFD.tStart = t  # local t and not account for scr refresh
                trainVideoAFD.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trainVideoAFD, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trainVideoAFD.started')
                # update status
                trainVideoAFD.status = STARTED
                trainVideoAFD.setAutoDraw(True)
                trainVideoAFD.play()
            
            # if trainVideoAFD is stopping this frame...
            if trainVideoAFD.status == STARTED:
                if bool(False) or trainVideoAFD.isFinished:
                    # keep track of stop time/frame for later
                    trainVideoAFD.tStop = t  # not accounting for scr refresh
                    trainVideoAFD.tStopRefresh = tThisFlipGlobal  # on global time
                    trainVideoAFD.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trainVideoAFD.stopped')
                    # update status
                    trainVideoAFD.status = FINISHED
                    trainVideoAFD.setAutoDraw(False)
                    trainVideoAFD.stop()
            if trainVideoAFD.isFinished:  # force-end the Routine
                continueRoutine = False
            
            # *trainLabelAFD* updates
            
            # if trainLabelAFD is starting this frame...
            if trainLabelAFD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trainLabelAFD.frameNStart = frameN  # exact frame index
                trainLabelAFD.tStart = t  # local t and not account for scr refresh
                trainLabelAFD.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trainLabelAFD, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trainLabelAFD.started')
                # update status
                trainLabelAFD.status = STARTED
                trainLabelAFD.setAutoDraw(True)
            
            # if trainLabelAFD is active this frame...
            if trainLabelAFD.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[trainVideoAFD]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                afd5TrainRoutine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in afd5TrainRoutine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "afd5TrainRoutine" ---
        for thisComponent in afd5TrainRoutine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for afd5TrainRoutine
        afd5TrainRoutine.tStop = globalClock.getTime(format='float')
        afd5TrainRoutine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('afd5TrainRoutine.stopped', afd5TrainRoutine.tStop)
        trainVideoAFD.stop()  # ensure movie has stopped at end of Routine
        # the Routine "afd5TrainRoutine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'afd5TrainLoop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "afdinstructions" ---
    # create an object to store info about Routine afdinstructions
    afdinstructions = data.Routine(
        name='afdinstructions',
        components=[key_resp_instruction_3, instruction_text_3],
    )
    afdinstructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instruction_3
    key_resp_instruction_3.keys = []
    key_resp_instruction_3.rt = []
    _key_resp_instruction_3_allKeys = []
    # store start times for afdinstructions
    afdinstructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    afdinstructions.tStart = globalClock.getTime(format='float')
    afdinstructions.status = STARTED
    thisExp.addData('afdinstructions.started', afdinstructions.tStart)
    afdinstructions.maxDuration = None
    # keep track of which components have finished
    afdinstructionsComponents = afdinstructions.components
    for thisComponent in afdinstructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "afdinstructions" ---
    afdinstructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_instruction_3* updates
        waitOnFlip = False
        
        # if key_resp_instruction_3 is starting this frame...
        if key_resp_instruction_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instruction_3.frameNStart = frameN  # exact frame index
            key_resp_instruction_3.tStart = t  # local t and not account for scr refresh
            key_resp_instruction_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instruction_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_instruction_3.started')
            # update status
            key_resp_instruction_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_instruction_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_instruction_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_instruction_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_instruction_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instruction_3_allKeys.extend(theseKeys)
            if len(_key_resp_instruction_3_allKeys):
                key_resp_instruction_3.keys = _key_resp_instruction_3_allKeys[-1].name  # just the last key pressed
                key_resp_instruction_3.rt = _key_resp_instruction_3_allKeys[-1].rt
                key_resp_instruction_3.duration = _key_resp_instruction_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instruction_text_3* updates
        
        # if instruction_text_3 is starting this frame...
        if instruction_text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_text_3.frameNStart = frameN  # exact frame index
            instruction_text_3.tStart = t  # local t and not account for scr refresh
            instruction_text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_text_3.started')
            # update status
            instruction_text_3.status = STARTED
            instruction_text_3.setAutoDraw(True)
        
        # if instruction_text_3 is active this frame...
        if instruction_text_3.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            afdinstructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in afdinstructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "afdinstructions" ---
    for thisComponent in afdinstructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for afdinstructions
    afdinstructions.tStop = globalClock.getTime(format='float')
    afdinstructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('afdinstructions.stopped', afdinstructions.tStop)
    # check responses
    if key_resp_instruction_3.keys in ['', [], None]:  # No response was made
        key_resp_instruction_3.keys = None
    thisExp.addData('key_resp_instruction_3.keys',key_resp_instruction_3.keys)
    if key_resp_instruction_3.keys != None:  # we had a response
        thisExp.addData('key_resp_instruction_3.rt', key_resp_instruction_3.rt)
        thisExp.addData('key_resp_instruction_3.duration', key_resp_instruction_3.duration)
    thisExp.nextEntry()
    # the Routine "afdinstructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    afd5TestLoop = data.TrialHandler2(
        name='afd5TestLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('afd5_testing_conditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(afd5TestLoop)  # add the loop to the experiment
    thisAfd5TestLoop = afd5TestLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisAfd5TestLoop.rgb)
    if thisAfd5TestLoop != None:
        for paramName in thisAfd5TestLoop:
            globals()[paramName] = thisAfd5TestLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisAfd5TestLoop in afd5TestLoop:
        currentLoop = afd5TestLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisAfd5TestLoop.rgb)
        if thisAfd5TestLoop != None:
            for paramName in thisAfd5TestLoop:
                globals()[paramName] = thisAfd5TestLoop[paramName]
        
        # --- Prepare to start Routine "afd5TestRoutine" ---
        # create an object to store info about Routine afd5TestRoutine
        afd5TestRoutine = data.Routine(
            name='afd5TestRoutine',
            components=[testVideoAFD, promptAFD, testResponseAFD],
        )
        afd5TestRoutine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        testVideoAFD.setMovie(video_file)
        # create starting attributes for testResponseAFD
        testResponseAFD.keys = []
        testResponseAFD.rt = []
        _testResponseAFD_allKeys = []
        # store start times for afd5TestRoutine
        afd5TestRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        afd5TestRoutine.tStart = globalClock.getTime(format='float')
        afd5TestRoutine.status = STARTED
        thisExp.addData('afd5TestRoutine.started', afd5TestRoutine.tStart)
        afd5TestRoutine.maxDuration = None
        # keep track of which components have finished
        afd5TestRoutineComponents = afd5TestRoutine.components
        for thisComponent in afd5TestRoutine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "afd5TestRoutine" ---
        # if trial has changed, end Routine now
        if isinstance(afd5TestLoop, data.TrialHandler2) and thisAfd5TestLoop.thisN != afd5TestLoop.thisTrial.thisN:
            continueRoutine = False
        afd5TestRoutine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *testVideoAFD* updates
            
            # if testVideoAFD is starting this frame...
            if testVideoAFD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                testVideoAFD.frameNStart = frameN  # exact frame index
                testVideoAFD.tStart = t  # local t and not account for scr refresh
                testVideoAFD.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(testVideoAFD, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'testVideoAFD.started')
                # update status
                testVideoAFD.status = STARTED
                testVideoAFD.setAutoDraw(True)
                testVideoAFD.play()
            
            # if testVideoAFD is stopping this frame...
            if testVideoAFD.status == STARTED:
                if bool(False) or testVideoAFD.isFinished:
                    # keep track of stop time/frame for later
                    testVideoAFD.tStop = t  # not accounting for scr refresh
                    testVideoAFD.tStopRefresh = tThisFlipGlobal  # on global time
                    testVideoAFD.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'testVideoAFD.stopped')
                    # update status
                    testVideoAFD.status = FINISHED
                    testVideoAFD.setAutoDraw(False)
                    testVideoAFD.stop()
            if testVideoAFD.isFinished:  # force-end the Routine
                continueRoutine = False
            
            # *promptAFD* updates
            
            # if promptAFD is starting this frame...
            if promptAFD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                promptAFD.frameNStart = frameN  # exact frame index
                promptAFD.tStart = t  # local t and not account for scr refresh
                promptAFD.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(promptAFD, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'promptAFD.started')
                # update status
                promptAFD.status = STARTED
                promptAFD.setAutoDraw(True)
            
            # if promptAFD is active this frame...
            if promptAFD.status == STARTED:
                # update params
                pass
            
            # *testResponseAFD* updates
            waitOnFlip = False
            
            # if testResponseAFD is starting this frame...
            if testResponseAFD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                testResponseAFD.frameNStart = frameN  # exact frame index
                testResponseAFD.tStart = t  # local t and not account for scr refresh
                testResponseAFD.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(testResponseAFD, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'testResponseAFD.started')
                # update status
                testResponseAFD.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(testResponseAFD.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(testResponseAFD.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if testResponseAFD.status == STARTED and not waitOnFlip:
                theseKeys = testResponseAFD.getKeys(keyList=['1', '2', '3', '4', '5'], ignoreKeys=["escape"], waitRelease=False)
                _testResponseAFD_allKeys.extend(theseKeys)
                if len(_testResponseAFD_allKeys):
                    testResponseAFD.keys = _testResponseAFD_allKeys[-1].name  # just the last key pressed
                    testResponseAFD.rt = _testResponseAFD_allKeys[-1].rt
                    testResponseAFD.duration = _testResponseAFD_allKeys[-1].duration
                    # was this correct?
                    if (testResponseAFD.keys == str(correct_ans)) or (testResponseAFD.keys == correct_ans):
                        testResponseAFD.corr = 1
                    else:
                        testResponseAFD.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[testVideoAFD]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                afd5TestRoutine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in afd5TestRoutine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "afd5TestRoutine" ---
        for thisComponent in afd5TestRoutine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for afd5TestRoutine
        afd5TestRoutine.tStop = globalClock.getTime(format='float')
        afd5TestRoutine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('afd5TestRoutine.stopped', afd5TestRoutine.tStop)
        testVideoAFD.stop()  # ensure movie has stopped at end of Routine
        # check responses
        if testResponseAFD.keys in ['', [], None]:  # No response was made
            testResponseAFD.keys = None
            # was no response the correct answer?!
            if str(correct_ans).lower() == 'none':
               testResponseAFD.corr = 1;  # correct non-response
            else:
               testResponseAFD.corr = 0;  # failed to respond (incorrectly)
        # store data for afd5TestLoop (TrialHandler)
        afd5TestLoop.addData('testResponseAFD.keys',testResponseAFD.keys)
        afd5TestLoop.addData('testResponseAFD.corr', testResponseAFD.corr)
        if testResponseAFD.keys != None:  # we had a response
            afd5TestLoop.addData('testResponseAFD.rt', testResponseAFD.rt)
            afd5TestLoop.addData('testResponseAFD.duration', testResponseAFD.duration)
        # Run 'End Routine' code from scoreAFD
        if testResponseAFD.keys == str(correct_ans):
            correct = 1
        else:
            correct = 0
        thisExp.addData('correct', correct)
        thisExp.addData('given_answer', testResponseAFD.keys)
        
        # the Routine "afd5TestRoutine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'afd5TestLoop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "goodbyeRoutine" ---
    # create an object to store info about Routine goodbyeRoutine
    goodbyeRoutine = data.Routine(
        name='goodbyeRoutine',
        components=[goodbyetext, key_resp],
    )
    goodbyeRoutine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for goodbyeRoutine
    goodbyeRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    goodbyeRoutine.tStart = globalClock.getTime(format='float')
    goodbyeRoutine.status = STARTED
    thisExp.addData('goodbyeRoutine.started', goodbyeRoutine.tStart)
    goodbyeRoutine.maxDuration = None
    # keep track of which components have finished
    goodbyeRoutineComponents = goodbyeRoutine.components
    for thisComponent in goodbyeRoutine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "goodbyeRoutine" ---
    goodbyeRoutine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *goodbyetext* updates
        
        # if goodbyetext is starting this frame...
        if goodbyetext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            goodbyetext.frameNStart = frameN  # exact frame index
            goodbyetext.tStart = t  # local t and not account for scr refresh
            goodbyetext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(goodbyetext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'goodbyetext.started')
            # update status
            goodbyetext.status = STARTED
            goodbyetext.setAutoDraw(True)
        
        # if goodbyetext is active this frame...
        if goodbyetext.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            goodbyeRoutine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in goodbyeRoutine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "goodbyeRoutine" ---
    for thisComponent in goodbyeRoutine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for goodbyeRoutine
    goodbyeRoutine.tStop = globalClock.getTime(format='float')
    goodbyeRoutine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('goodbyeRoutine.stopped', goodbyeRoutine.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    # Run 'End Routine' code from summaryCode
    # Compute overall test accuracy
    ucf5_acc = ucf5TestLoop.data['correct'].mean() * 100
    afd5_acc = afd5TestLoop.data['correct'].mean() * 100
    
    # Save to output
    thisExp.addData('ucf5_accuracy_pct', round(ucf5_acc, 1))
    thisExp.addData('afd5_accuracy_pct', round(afd5_acc, 1))
    thisExp.nextEntry()
    # the Routine "goodbyeRoutine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
