clear; clc; close all;

numFiles = 1;

for i = 1:numFiles
    
    %% 1. אתחול וטעינת הקובץ
    inputFilename = sprintf('visente_external_clock.mat', i);
    outputAudioFile = sprintf('recovered_speech_%d.wav', i);

    if ~isfile(inputFilename)
        warning('File %s not found! Skipping...', inputFilename);
        continue;
    end

    disp(['Loading file: ', inputFilename, '...']);
    data = load(inputFilename);

    videoData = data.vidImages;  
    realFPS = data.fps;          

    fprintf('Loaded video %d successfully.\n', i);
    fprintf('Frames: %d\n', size(videoData,3));
    fprintf('Resolution: %dx%d\n', size(videoData,1), size(videoData,2));
    fprintf('Sample Rate (FPS): %.2f Hz\n', realFPS);

    %% 2. הרצת האלגוריתם
    disp('Running Visual Microphone Algorithm...');

    [pos, dpos, L, corrValues, corrImages] = Frames2Sound(videoData, 'TRUE', 'FALSE', 1);

    disp('Algorithm finished.');

    %% 3. עיבוד האות
    rawSignal = dpos(1, :);

    cutoffFreq = 100;
    if realFPS > cutoffFreq * 2
        cleanSignal = highpass(rawSignal, cutoffFreq, realFPS);
    else
        cleanSignal = rawSignal;
        warning('FPS too low for High-pass filter.');
    end

    maxVal = max(abs(cleanSignal));
    if maxVal > 0
        normalizedSignal = cleanSignal / maxVal;
    else
        normalizedSignal = cleanSignal;
        warning('Signal is empty.');
    end

    %% 4. שמירה והשמעה
    intFPS = round(realFPS);
    audiowrite(outputAudioFile, normalizedSignal, intFPS);

    fprintf('Saved: %s\n', outputAudioFile);

    % אפשר לבטל השמעה אם יש הרבה קבצים
    % sound(normalizedSignal, realFPS);

    %% 5. גרף (אופציונלי לכל קובץ)
    figure('Name', ['Result File ', num2str(i)], 'NumberTitle', 'off');

    subplot(2,1,1);
    t = (0:length(normalizedSignal)-1) / realFPS;
    plot(t, normalizedSignal);
    title(['Recovered Signal - File ', num2str(i)]);
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on;

    subplot(2,1,2);
    spectrogram(normalizedSignal, 512, 256, 512, realFPS, 'yaxis');
    title('Spectrogram');

end

disp('All files processed.');