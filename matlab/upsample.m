% Set the folder where the recordings are located
folderPath = 'C:\Users\oror8\Downloads\ForSync\ForSync'; 
targetFs = 16000; 

% Get a list of ALL .wav files in the directory
wavFiles = dir(fullfile(folderPath, '*.wav'));

% Loop through dynamically found files
for k = 1:length(wavFiles)
    % Get the current file name
    fileName = wavFiles(k).name;
    
    % Skip files that already have "_16kHz" in their name to avoid double processing
    if contains(fileName, '_16kHz')
        continue;
    end
    
    fullFilePath = fullfile(folderPath, fileName);
    
    try
        % Load the original WAV file
        [inputAudio, originalFs] = audioread(fullFilePath);
        
        if originalFs ~= targetFs
            [P, Q] = rat(targetFs / originalFs);
            outputAudio = resample(inputAudio, P, Q);
        else
            outputAudio = inputAudio;
        end
        
        % Create new name: insert "_16kHz" before the ".wav" extension
        [~, name, ext] = fileparts(fileName);
        outputFileName = sprintf('%s_16kHz%s', name, ext);
        outputFullFilePath = fullfile(folderPath, outputFileName);
        
        % Write the file
        audiowrite(outputFullFilePath, outputAudio, targetFs);
        disp(['Processed: ', fileName, ' -> ', outputFileName]);
        
    catch ME
        disp(['Error processing ', fileName, ': ', ME.message]);
    end
end