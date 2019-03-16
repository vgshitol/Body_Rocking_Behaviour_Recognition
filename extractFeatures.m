wristFile = fopen('Session05/wristIMU.txt','r');
armFile = fopen('Session05/armIMU.txt','r');
%featureFile = fopen('Session01/features.csv','w');
formatSpec = '%f %f %f %f %f %f';
sizeA = [6 Inf];
wristData = fscanf(wristFile,formatSpec,sizeA);
armData = fscanf(armFile,formatSpec,sizeA);
wristData_padded = [zeros(6,199) wristData];
armData_padded = [zeros(6,199) armData];
tot_length = length(wristData);
features = zeros(tot_length,84);
for i = 1:tot_length
    a_wrist = wristData_padded(1:3,i:i+199);
    g_wrist = wristData_padded(4:6,i:i+199);
    a_wrist_feature = m_getFeaturesIMU(a_wrist', 50);
    g_wrist_feature = m_getFeaturesIMU(g_wrist', 50);
    
    a_arm = armData_padded(1:3,i:i+199);
    g_arm = armData_padded(4:6,i:i+199);
    a_arm_feature = m_getFeaturesIMU(a_arm', 50);
    g_arm_feature = m_getFeaturesIMU(g_arm', 50);
    
    features(i,:) = [a_wrist_feature g_wrist_feature a_arm_feature g_arm_feature];
end
csvwrite('Session05/features.csv',features);




