function F = m_getFeaturesIMU(x,fs)
% F = m_getFeaturesIMU(x,fs)
%
% This function computes some temporal and frequency based features from
% IMU signals. It is assumed that x is of size N x 3, where N is the number of
% samples in the window, and 3 is because we are considering accelleration or
% gyroscope alone. The 'fs' is the nominal frequency of sampling used for this
% device.

% Computing temporal features
f1 = mean(x,1); % Means
C = cov(x);
f2 = [C(1,1:3) C(2,2:3) C(3,3)]; % Variances and Covariances
f3 = [skewness(x(:,1)) skewness(x(:,2)) skewness(x(:,3))]; % Skewness
f4 = [kurtosis(x(:,1)) kurtosis(x(:,2)) kurtosis(x(:,3))]; % Kurtosis

% Computing frequency features (max freq response)
f5 = zeros(1,3);
f6 = zeros(1,3);
for i = 1:3
    g = abs(fft(x(:,i)));
    g = g(1:round(length(g)/2));
    g(1) = 0;
    w = fs*(0:length(g))/(2*length(g));
    [v,idx] = max(g);
    f5(i) = v;
    f6(i) = w(idx);
end

% Putting together feature vector
F = [f1 f2 f3 f4 f5 f6];
