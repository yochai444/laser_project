%% extract_variants.m
% -------------------------------------------------------------------------
% Produces six WAV variants from ONE .mat recording, to test two hypotheses
% about why the optical channel loses 29 points against the microphone:
%
%   AXIS. matrix2sound.m uses rawSignal = dpos(1,:), the X axis only.
%   Frames2Sound returns a two-dimensional displacement, so if the geometry of
%   the laser, camera and surface put the dominant vibration closer to Y in some
%   sessions, that session's signal was largely discarded.
%
%   HIGHPASS. The cutoff is 100 Hz. Measured mic-to-laser coherence at 125 Hz
%   was 0.246 for speaker 07 and 0.130 for speaker 01, so there is signal being
%   removed. Male fundamental frequency runs 85-155 Hz, and the coherence peak
%   for this corpus sits at 188-312 Hz - right above the cut.
%
% The two are varied independently (3 axis choices x 2 cutoffs) so their
% contributions can be told apart rather than confounded.
%
% Frames2Sound is the expensive part and runs only once; all six variants are
% derived from the same dpos. Output is resampled to 16 kHz to match the corpus.
%
% USAGE
%   edit inputFile below, then run. Then, in Python:
%       python compare_variants.py --variant-dir ./variants \
%           --mic ./microphone/microphone_017.wav
% -------------------------------------------------------------------------

clear; clc; close all;

inputFile  = 'matrix_017.mat';   % speaker 01; we already have its spectrogram
outputDir  = 'variants';
targetSR   = 16000;              % match the rest of the corpus
BAND       = [60 700];           % where the projection is chosen to maximise energy

if ~isfile(inputFile)
    error('File %s not found.', inputFile);
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% 1. Load and run the tracker once
fprintf('Loading %s ...\n', inputFile);
data     = load(inputFile);
vid      = data.vidImages;
realFPS  = double(data.fps);
intFPS   = round(realFPS);

fprintf('  %d frames, %dx%d, %.2f fps (Nyquist %.0f Hz)\n', ...
    size(vid,3), size(vid,1), size(vid,2), realFPS, realFPS/2);

cacheFile = fullfile(outputDir, ['dpos_' inputFile]);
if isfile(cacheFile)
    fprintf('Reusing cached displacement from %s\n', cacheFile);
    load(cacheFile, 'dpos');
else
    fprintf('Running Frames2Sound (this is the slow step) ...\n');
    [~, dpos, ~, ~, ~] = Frames2Sound(vid, 'TRUE', 'FALSE', 1);
    save(cacheFile, 'dpos');
end

x = dpos(1,:);
y = dpos(2,:);

%% 2. Choose the projection direction
% The direction is picked to maximise energy inside the speech band rather than
% total variance. Total variance is dominated by low-frequency drift, which
% would steer the projection towards mechanical movement instead of speech.
xb = bandLimit(x, realFPS, BAND);
yb = bandLimit(y, realFPS, BAND);

C = cov([xb(:), yb(:)]);
[V, D] = eig(C);
[~, k] = max(diag(D));
w = V(:,k);
if w(1) < 0, w = -w; end        % sign is arbitrary; fix it for reproducibility

proj = w(1)*x + w(2)*y;

fprintf('\nProjection direction: %.3f * X + %.3f * Y  (angle %.1f deg from X)\n', ...
    w(1), w(2), atan2d(w(2), w(1)));
fprintf('In-band energy: X %.3f | Y %.3f | projection %.3f  (relative units)\n', ...
    sum(xb.^2), sum(yb.^2), sum(bandLimit(proj, realFPS, BAND).^2));
if abs(w(2)) > abs(w(1))
    fprintf('  -> the dominant axis is Y. matrix2sound.m was reading the weaker one.\n');
end

%% 3. Build the six variants
axes   = {x,     y,     proj};
axNames= {'X',   'Y',   'PCA'};
cutoffs= [100, 60];

fprintf('\n%-14s %10s %10s %10s\n', 'variant', 'in-band', 'total', 'ratio');
for a = 1:numel(axes)
    for c = 1:numel(cutoffs)
        fc   = cutoffs(c);
        name = sprintf('%s_hp%d', axNames{a}, fc);

        sig = axes{a};
        if realFPS > 2*fc
            sig = highpass(sig, fc, realFPS);
        else
            warning('FPS too low for a %d Hz highpass; skipping the filter.', fc);
        end

        % Resample to 16 kHz so the file drops straight into the existing
        % tooling. This adds no information above the laser Nyquist; it only
        % makes the variants directly comparable with the corpus.
        out = resample(double(sig(:)), targetSR, intFPS);

        m = max(abs(out));
        if m > 0, out = out / m; end

        inBand = sum(bandLimit(out, targetSR, [150 700]).^2);
        total  = sum(out.^2);
        fprintf('%-14s %10.1f %10.1f %10.3f\n', name, inBand, total, inBand/total);

        audiowrite(fullfile(outputDir, ['variant_' name '.wav']), out, targetSR);
    end
end

fprintf('\nWrote 6 WAV files to ./%s\n', outputDir);

%% 4. Spectrograms, for a quick visual read
figure('Name','Variant comparison','NumberTitle','off','Position',[50 50 1400 800]);
i = 0;
for a = 1:numel(axes)
    for c = 1:numel(cutoffs)
        i = i + 1;
        name = sprintf('%s_hp%d', axNames{a}, cutoffs(c));
        [s, fs] = audioread(fullfile(outputDir, ['variant_' name '.wav']));
        subplot(3, 2, i);
        spectrogram(s, 1024, 768, 1024, fs, 'yaxis');
        ylim([0 1]);                     % 0-1000 Hz; yaxis is in kHz
        title(name, 'Interpreter', 'none');
        colorbar('off');
    end
end
sgtitle('Six extractions of the same recording, 0-1000 Hz');

%% ------------------------------------------------------------------------
function out = bandLimit(sig, fs, band)
% Band-pass helper that degrades gracefully when the band exceeds Nyquist.
    lo = band(1);
    hi = min(band(2), fs/2 * 0.98);
    if fs <= 2*lo
        out = sig(:);
        return;
    end
    [b, a] = butter(4, [lo hi] / (fs/2), 'bandpass');
    out = filtfilt(b, a, double(sig(:)));
end
