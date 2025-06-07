function [obj,recal_index] = fitness(~)

    system('C:\Users\le_lianxiang\Desktop\pca_aircompressor_optimization\pca_ada_fuben\run_python.bat');
    % 打开文件
    fid = fopen('calculation_status.txt', 'r');
    if fid == -1
        error('文件打开失败');
    end
    % 使用 textscan 读取文件内容，按空格分割
    data = textscan(fid, '%f %f');  % 读取两列字符串
    fclose(fid);
    % 获取数据
    recal_index = data{1}(1);  % 第一列
    obj = data{2}(1);                 % 第二列

    % 示例：打印读取的所有字符串
    fprintf('%.5e\n',obj);
end