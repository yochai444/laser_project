%% region_tracking.m
% -------------------------------------------------------------------------
% Tests whether the global displacement estimate discards information that is
% present in different parts of the image.
%
% THE CLAIM BEING TESTED
%   Frames2Sound correlates the whole 64x64 frame against the previous one and
%   returns ONE displacement vector per frame. If the surface vibrates in
%   spatial modes, different regions carry different information and averaging
%   them cancels part of it. That is a claim about the raw frames containing
%   more than the current extraction delivers, and it has never been checked.
%
% WHY IT IS WORTH CHECKING
%   Measured microphone-to-laser coherence is 0.42-0.46 for speakers 06-07 and
%   0.22-0.31 for 01-05. Same code, same algorithm, a factor of two apart. An
%   estimator that sensitive to conditions usually has room in it.
%
% WHAT THIS DOES NOT ADDRESS
%   Coherence above 700 Hz measured 0.006 across every estimator setting tried,
%   and 1467 Hz is the Nyquist limit of a 2940 fps camera. Nothing here can
%   reach that range. The ceiling this could move is the 11 points between the
%   laser at 67% and the microphone at 78% within the band that already works;
%   the 22 points above it need a faster camera.
%
% METHOD
%   The frame is cut into a 3x3 grid with overlap, and Frames2Sound runs on each
%   tile separately. That gives 9 tiles x 2 axes = 18 channels instead of 2. Each
%   is written as its own WAV so band_coherence.py can score them individually
%   against the microphone, and two combinations are written as well: the mean,
%   and a coherence-weighted sum.
%
% COST
%   Frames2Sound runs 9 times on tiles of about a quarter the area, so expect
%   roughly 2-4x the time of one full-frame run.
%
% USAGE
%   Edit inputFile, run, then:
%     python compare_variants.py --variant-dir ./region_variants \
%         --mic ./microphone/microphone_017.wav
% -------------------------------------------------------------------------

clear; clc; close all;

inputFile  = 'matrix_017.mat';
outputDir  = 'region_variants';
targetSR   = 16000;
GRID       = 3;        % 3x3 tiles
OVERLAP    = 0.25;     % tiles overlap by this fraction, so edges are covered

if ~isfile(inputFile), error('File %s not found.', inputFile); end
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

fprintf('Loading %s ...\n', inputFile);
data    = load(inputFile);
vid     = data.vidImages;
realFPS = double(data.fps);
intFPS  = round(realFPS);
[H, W, N] = size(vid);
fprintf('  %d frames, %dx%d, %.1f fps\n', N, H, W, realFPS);

% Tile geometry
tile   = floor(H / (GRID - (GRID-1)*OVERLAP));
step   = floor(tile * (1 - OVERLAP));
starts = 1 : step : (H - tile + 1);
starts = starts(1:min(GRID, numel(starts)));
fprintf('  %dx%d tiles of %dx%d, step %d\n', ...
    numel(starts), numel(starts), tile, tile, step);

if tile < 24
    warning(['Tiles are %dx%d. Speckle correlation gets noisy on small ' ...
             'patches; if every tile scores worse than the full frame, that ' ...
             'is the likely reason rather than an absence of spatial ' ...
             'structure.'], tile, tile);
end

%% Full-frame reference, so there is something to compare against
fprintf('\n[reference] full frame\n');
cacheFile = fullfile(outputDir, 'dpos_full.mat');
if isfile(cacheFile)
    fprintf('  reusing cache\n');
    load(cacheFile, 'dposFull');
else
    [~, dposFull, ~, ~, ~] = Frames2Sound(vid, 'FALSE', 'FALSE', 'TRUE', 'nofigs');
    save(cacheFile, 'dposFull');
end
writeSignal(dposFull(1,:), 'full_X', outputDir, intFPS, targetSR);
writeSignal(dposFull(2,:), 'full_Y', outputDir, intFPS, targetSR);
writeSignal(project(dposFull, realFPS), 'full_PCA', outputDir, intFPS, targetSR);

%% Per-tile tracking
nTiles = numel(starts)^2;
allSig = zeros(nTiles*2, size(dposFull, 2));
row = 0;
k = 0;

for i = 1:numel(starts)
    for j = 1:numel(starts)
        k = k + 1;
        r0 = starts(i); c0 = starts(j);
        label = sprintf('tile%d%d', i, j);
        fprintf('\n[%d/%d] %s  rows %d-%d, cols %d-%d\n', ...
            k, nTiles, label, r0, r0+tile-1, c0, c0+tile-1);

        cacheFile = fullfile(outputDir, ['dpos_' label '.mat']);
        if isfile(cacheFile)
            fprintf('  reusing cache\n');
            load(cacheFile, 'dposT');
        else
            sub = vid(r0:r0+tile-1, c0:c0+tile-1, :);
            [~, dposT, ~, ~, ~] = Frames2Sound(sub, 'FALSE', 'FALSE', 'TRUE', 'nofigs');
            save(cacheFile, 'dposT');
        end

        n = min(size(dposT,2), size(allSig,2));
        allSig(row+1, 1:n) = dposT(1, 1:n);
        allSig(row+2, 1:n) = dposT(2, 1:n);
        row = row + 2;

        writeSignal(dposT(1,:), [label '_X'], outputDir, intFPS, targetSR);
        writeSignal(dposT(2,:), [label '_Y'], outputDir, intFPS, targetSR);
    end
end

%% Two ways of combining the tiles
% Plain mean. If the tiles all carry the same thing, this is what the global
% estimate already computes and it should score about the same.
writeSignal(mean(allSig, 1), 'combo_mean', outputDir, intFPS, targetSR);

% Energy-weighted first principal component across all 18 channels. If the tiles
% carry DIFFERENT things, a learned linear combination should beat both the mean
% and the full-frame estimate. That difference is the whole question.
B = zeros(size(allSig));
for r = 1:size(allSig,1)
    B(r,:) = bandLimit(allSig(r,:), realFPS, [100 1400]);
end
C = cov(B');
[V, D] = eig(C);
[~, kk] = max(diag(D));
w = V(:,kk); if sum(w) < 0, w = -w; end
writeSignal(w' * allSig, 'combo_pca', outputDir, intFPS, targetSR);

fprintf('\nTile weights in the combined signal:\n');
for r = 1:2:numel(w)
    fprintf('  tile %d: X %+.3f  Y %+.3f\n', (r+1)/2, w(r), w(r+1));
end

fprintf('\nWrote WAVs to ./%s\n', outputDir);
fprintf(['\nScore them against the microphone:\n' ...
         '  python compare_variants.py --variant-dir ./%s \\\n' ...
         '      --mic ./microphone/microphone_017.wav\n\n' ...
         'What the answer means:\n' ...
         '  every tile below full_PCA      -> the vibration is uniform, the\n' ...
         '                                    global average loses nothing,\n' ...
         '                                    and this direction is closed\n' ...
         '  combo_pca clearly above full_PCA -> tiles carry different\n' ...
         '                                    information and re-extracting\n' ...
         '                                    the corpus this way is worth it\n'], outputDir);

%% ------------------------------------------------------------------------
function s = project(dpos, fs)
% Combine the two axes along the direction with most energy in 100-1400 Hz.
    xb = bandLimit(dpos(1,:), fs, [100 1400]);
    yb = bandLimit(dpos(2,:), fs, [100 1400]);
    C = cov([xb(:), yb(:)]);
    [V, D] = eig(C);
    [~, k] = max(diag(D));
    w = V(:,k); if w(1) < 0, w = -w; end
    s = w(1)*dpos(1,:) + w(2)*dpos(2,:);
end

function writeSignal(sig, label, outputDir, intFPS, targetSR)
% Resample to 16 kHz and write 32-bit float, with no highpass. Per-frequency
% coherence is invariant to linear filtering, so filtering here could only
% remove something before it is looked at.
    out = resample(double(sig(:)), targetSR, intFPS);
    m = max(abs(out));
    if m > 0, out = out / m; end
    audiowrite(fullfile(outputDir, ['variant_' label '.wav']), out, targetSR, ...
        'BitsPerSample', 32);
end

function out = bandLimit(sig, fs, band)
    lo = band(1);
    hi = min(band(2), fs/2 * 0.98);
    if fs <= 2*lo || hi <= lo
        out = sig(:).'; return;
    end
    [b, a] = butter(4, [lo hi] / (fs/2), 'bandpass');
    out = filtfilt(b, a, double(sig(:))).';
end
