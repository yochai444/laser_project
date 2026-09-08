%% estimator_sweep.m
% -------------------------------------------------------------------------
% Tests whether the 700-1467 Hz range can be recovered from the existing
% recordings, without new hardware.
%
% WHY THIS RANGE
%   The camera runs at ~2940 fps, so Nyquist is ~1467 Hz. Measured microphone-
%   to-laser coherence peaks at 190-310 Hz and collapses past 700 Hz. Everything
%   between 700 and 1467 Hz was physically sampled and is still coming back
%   empty, so something in the extraction is losing it rather than physics.
%
%   The full-band experiment put a number on what that costs: with 50-7500 Hz
%   the same model reached 100% on an unseen speaker, against 77.8% restricted
%   to 156-688 Hz.
%
% WHY ONLY THE ESTIMATOR FLAGS ARE VARIED
%   Per-frequency coherence is invariant to linear filtering: attenuating a band
%   scales signal and noise together and the correlation does not move. That is
%   why the earlier 60 Hz versus 100 Hz highpass comparison returned identical
%   numbers to three decimals. lenconv, the highpass and the cumulative sum are
%   all linear, so none of them can explain the collapse and none of them can
%   fix it.
%
%   What can change is the ESTIMATOR - how the displacement is measured. The
%   three arguments to Frames2Sound each alter how much noise enters that
%   estimate at each frequency:
%
%     bkg_removal          subtracts a 128-frame running background
%     noFarting            restricts the correlation search to a small mask
%     subpixelExponential  applies a Gaussian to the correlation before the
%                          parabolic subpixel fit
%
%   matrix2sound.m uses ('TRUE','FALSE',1) and nothing else was ever tried.
%
% COST
%   Frames2Sound runs once per combination, which is the slow part. Six
%   combinations on one file is roughly 30-90 minutes.
%
% USAGE
%   Edit inputFile, run, then in Python:
%     python compare_variants.py --variant-dir ./est_variants \
%         --mic ./microphone/microphone_017.wav
% -------------------------------------------------------------------------

clear; clc; close all;

inputFile  = 'matrix_017.mat';
outputDir  = 'est_variants';
targetSR   = 16000;

if ~isfile(inputFile), error('File %s not found.', inputFile); end
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

fprintf('Loading %s ...\n', inputFile);
data    = load(inputFile);
vid     = data.vidImages;
realFPS = double(data.fps);
intFPS  = round(realFPS);
fprintf('  %d frames, %.1f fps, Nyquist %.0f Hz\n', ...
    size(vid,3), realFPS, realFPS/2);

%% Combinations to try
% Columns: bkg_removal, noFarting, subpixelExponential, label
combos = {
    'TRUE',  'FALSE', 1, 'baseline'      % what matrix2sound.m uses today
    'TRUE',  'FALSE', 0, 'nosubpix'      % plain parabolic fit, no Gaussian
    'FALSE', 'FALSE', 1, 'nobkg'         % keep the raw frames
    'TRUE',  'TRUE',  1, 'masked'        % search near the previous peak only
    'FALSE', 'FALSE', 0, 'raw'           % neither
    'FALSE', 'TRUE',  1, 'nobkg_masked'
};

fprintf('\n%-14s %10s %10s %10s\n', 'variant', 'inband', 'hi700', 'hi/total');

for c = 1:size(combos, 1)
    bkg   = combos{c,1};
    fart  = combos{c,2};
    subp  = combos{c,3};
    label = combos{c,4};

    fprintf('\n[%d/%d] %s  (bkg=%s, mask=%s, subpix=%d)\n', ...
        c, size(combos,1), label, bkg, fart, subp);

    cacheFile = fullfile(outputDir, ['dpos_' label '.mat']);
    if isfile(cacheFile)
        fprintf('  reusing cache\n');
        load(cacheFile, 'pos', 'dpos');
    else
        subpArg = 'TRUE'; if subp == 0, subpArg = 'FALSE'; end
        [pos, dpos, ~, ~, ~] = Frames2Sound(vid, bkg, fart, subpArg, 'nofigs');
        save(cacheFile, 'pos', 'dpos');
    end

    % Both axes are kept. The projection is chosen to maximise energy in
    % 100-1400 Hz, which is the whole sampled range rather than the 60-700 Hz
    % used in the earlier axis test - the point here is what happens above 700.
    xb = bandLimit(dpos(1,:), realFPS, [100 1400]);
    yb = bandLimit(dpos(2,:), realFPS, [100 1400]);
    C  = cov([xb(:), yb(:)]);
    [V, D] = eig(C);
    [~, k] = max(diag(D));
    w = V(:,k); if w(1) < 0, w = -w; end
    sig = w(1)*dpos(1,:) + w(2)*dpos(2,:);

    % No highpass here. It is linear, so it cannot affect the measurement, and
    % leaving it out avoids removing anything before it can be looked at.
    out = resample(double(sig(:)), targetSR, intFPS);
    m = max(abs(out)); if m > 0, out = out / m; end

    % Written as 32-bit float rather than 16-bit integer. With a signal this
    % low-frequency dominated, content near 1 kHz can sit far below the peak,
    % and integer quantisation is one of the few non-linear steps in the chain.
    audiowrite(fullfile(outputDir, ['variant_' label '.wav']), out, targetSR, ...
        'BitsPerSample', 32);

    inband = sum(bandLimit(out, targetSR, [150 700]).^2);
    hi     = sum(bandLimit(out, targetSR, [700 1400]).^2);
    total  = sum(out.^2);
    fprintf('%-14s %10.1f %10.1f %10.4f\n', label, inband, hi, hi/total);
end

fprintf('\nWrote %d WAV files to ./%s\n', size(combos,1), outputDir);

%% The combination that was never actually tested
% Frames2Sound checks the background flag with
%
%     if (bkg_removal==1)
%
% but the caller passes the string 'TRUE'. In MATLAB 'TRUE'==1 compares the
% character codes [84 82 85 69] against 1 and returns [0 0 0 0], and an if on a
% vector of zeros is false. Background removal therefore never ran - not in the
% sweep above, not in matrix2sound.m, and not in any of the 369 extractions.
% That is why baseline, nobkg, masked and nobkg_masked returned identical
% numbers to four decimal places.
%
% Passing a numeric 1 takes the branch the comment in matrix2sound.m calls
% mandatory for vibrations. This is the one setting in the whole sweep that has
% not been seen before.

fprintf('\n[7/7] realbkg  (background removal actually enabled)\n');

cacheFile = fullfile(outputDir, 'dpos_realbkg.mat');
if isfile(cacheFile)
    fprintf('  reusing cache\n');
    load(cacheFile, 'pos', 'dpos');
else
    [pos, dpos, ~, ~, ~] = Frames2Sound(vid, 1, 'FALSE', 'TRUE', 'nofigs');
    save(cacheFile, 'pos', 'dpos');
end

xb = bandLimit(dpos(1,:), realFPS, [100 1400]);
yb = bandLimit(dpos(2,:), realFPS, [100 1400]);
C  = cov([xb(:), yb(:)]);
[V, D] = eig(C);
[~, k] = max(diag(D));
w = V(:,k); if w(1) < 0, w = -w; end
sig = w(1)*dpos(1,:) + w(2)*dpos(2,:);

out = resample(double(sig(:)), targetSR, intFPS);
m = max(abs(out)); if m > 0, out = out / m; end
audiowrite(fullfile(outputDir, 'variant_realbkg.wav'), out, targetSR, ...
    'BitsPerSample', 32);

inband = sum(bandLimit(out, targetSR, [150 700]).^2);
hi     = sum(bandLimit(out, targetSR, [700 1400]).^2);
fprintf('%-14s %10.1f %10.1f %10.4f\n', 'realbkg', inband, hi, hi/sum(out.^2));

fprintf(['\nThese energy numbers are only a rough guide. The measurement that\n' ...
         'decides the question is coherence against the microphone:\n' ...
         '  python compare_variants.py --variant-dir ./%s \\\n' ...
         '      --mic ./microphone/microphone_017.wav\n'], outputDir);

%% Spectrograms up to Nyquist, so the 700-1467 Hz range is visible
labels = [combos(:,4); {'realbkg'}];
figure('Name','Estimator sweep','NumberTitle','off','Position',[50 50 1400 900]);
for c = 1:numel(labels)
    label = labels{c};
    [s, fs] = audioread(fullfile(outputDir, ['variant_' label '.wav']));
    subplot(4, 2, c);
    spectrogram(s, 1024, 768, 1024, fs, 'yaxis');
    ylim([0 1.5]);                 % kHz; 1.467 is Nyquist
    hold on; yline(0.7, 'r--', 'LineWidth', 1);
    title(label, 'Interpreter', 'none');
    colorbar('off');
end
sgtitle('Red line at 700 Hz. Everything up to 1467 Hz was sampled.');

%% ------------------------------------------------------------------------
function out = bandLimit(sig, fs, band)
    lo = band(1);
    hi = min(band(2), fs/2 * 0.98);
    if fs <= 2*lo || hi <= lo
        out = sig(:); return;
    end
    [b, a] = butter(4, [lo hi] / (fs/2), 'bandpass');
    out = filtfilt(b, a, double(sig(:)));
end
