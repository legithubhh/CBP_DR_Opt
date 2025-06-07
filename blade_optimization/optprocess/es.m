% 基础参数设置
run_Maxtime = 10;       % 定义最大运行次数

%% 步骤1：创建列向量
error_s = (0:run_Maxtime-1)';  % 直接生成0-9的列向量
% 验证向量尺寸
assert(size(error_s,1) == run_Maxtime, '向量维度错误！');

%% 步骤2：创建结果目录
result_dir = fullfile(pwd, 'result');   % 获取当前路径并拼接目录名
if ~isfolder(result_dir)
    mkdir(result_dir);                  % 创建不存在的目录
end

%% 步骤3：文件写入操作
output_file = fullfile(result_dir, 'error statistics.txt');

% 使用低层文件I/O进行精确控制
fid = fopen(output_file, 'w');          % 以写入模式打开文件
if fid == -1
    error('文件创建失败，请检查权限！');  % 错误处理
end

% 格式化写入数据
for k = 1:numel(error_s)
    fprintf(fid, 'run_Maxtime%d: %d\n', run_Maxtime, error_s(k));
end

fclose(fid);                            % 必须关闭文件句柄

%% 可选验证步骤（调试用）
% type(output_file);  % 在命令窗口显示文件内容