filename = 'code\data\附件2.csv';

data = readtable(filename);
data = table2array(data);  % 将表格转换为数值数组

% 计算投影的吸收率中心 C
C = zeros(1, 180);  % 初始化吸收率中心数组
theta = linspace(0, 179, 180);  % 角度范围，theta 已经是角度制 (0 到 179)

% 计算吸收率中心 C_j
for j = 1:180
    p_j = data(:, j);  % 获取第 j 个投影数据
    C_j = (1:512) * p_j / sum(p_j);  % 计算吸收率中心
    C(j) = C_j;  % 存储每个投影的吸收率中心
end

% 1. 定义带偏移量的正弦函数
sin_func = @(params, theta) params(1) * sin(deg2rad(theta) + params(2)) + params(3);

% 2. 定义误差平方和函数
error_func = @(params) sum((C - sin_func(params, theta)).^2);

% 3. 初始化猜测参数
A_init = max(C) - min(C);  % 振幅初始值为数据最大值与最小值的差
phi_init = 0;  % 相位初始值为0
offset_init = mean(C);  % 偏移初始值为数据平均值

% 初始猜测参数
params_init = [A_init, phi_init, offset_init];

% 4. 使用 fminsearch 最小化误差平方和
options = optimset('Display', 'off');
params_opt = fminsearch(error_func, params_init, options);

% 拟合结果：振幅 A，相位 phi，偏移 offset
A = params_opt(1);  % 振幅
phi = params_opt(2);  % 相位
offset = params_opt(3);  % 偏移

% 显示拟合结果
fprintf('正弦拟合结果：A = %.4f, phi = %.4f, offset = %.4f\n', A, phi, offset);

% 5. 计算拟合的 C_fitted 值
C_fitted = sin_func(params_opt, theta);  % 使用拟合参数计算拟合曲线

% 绘制图像：原始数据与拟合结果
figure;
subplot(2,1,1);  % 第一个子图，显示计算的吸收率中心
plot(theta, C, 'b-', 'LineWidth', 2);  % 绘制计算的吸收率中心（蓝色实线）
hold on;
plot(theta, C_fitted, 'r--', 'LineWidth', 2);  % 绘制拟合的正弦曲线（红色虚线）
xlabel('投影角度 (\theta)');
ylabel('吸收率中心 C(\theta)');
title('投影吸收率中心轨迹与正弦拟合');
legend('计算的吸收率中心', '拟合的正弦曲线');
grid on;

% 可视化重建图像（反投影结果）
img_1 = iradon(data, theta, 512, 'Hann');  % 反投影重建
img_1 = medfilt2(img_1);  % 中值滤波降噪

subplot(2,1,2);  % 第二个子图，显示重建图像
imagesc(img_1);
axis image;
title('反投影重建图像');
colorbar;  % 添加颜色条

% 计算旋转中心坐标
% 假设吸收率中心坐标 (x_m, y_m) 已知
x_m = 50;  % 假设值，实际值需要通过问题1计算得到
y_m = 50;  % 假设值，实际值需要通过问题1计算得到

% 计算旋转中心坐标 (x_c, y_c)
R = A;  % 振幅即为旋转中心到吸收率中心的距离
alpha = -phi;  % 相位与旋转角度关联
x_c = x_m - R * cos(alpha);
y_c = y_m - R * sin(alpha);

% 输出旋转中心坐标
fprintf('旋转中心 (x_c, y_c) = (%.4f, %.4f) mm\n', x_c, y_c);
