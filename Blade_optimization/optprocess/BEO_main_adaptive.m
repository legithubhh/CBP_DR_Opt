%%============================================================
% BEO-SOP
% Bayesian Evolutionary Optimization on Single-objective Problems
% 初始化前设定范围，从初始化样本中确定自适应范围前提
% 精度判断标准
%%============================================================
function BEO_main_adaptive()
    %------------------Bayesian Evolutionary Optimization----------------------
    tic
    clc
    addpath(genpath(pwd)); % 将当前工作目录以及它的所有子目录添加到 MATLAB 的搜索路径
    mkdir('result');
    %Parameter setting
    D = 10; %控制变量数
    Nt = 10; %初始样本数,50
    FEmax = 200; % 最大评估样本数,500
    Np = 50; %代理模型评估样本数
    gmax = 20; %代理模型重复采样次数
    theta = 5.*ones(1,D);
    run_Maxtime = 5;
    nowbest = 1;% 总压损失比较值
    extension_ratio = 0.1;%调节每次边界扩展的幅度
    error_s = (0:run_Maxtime-1)';%储存每次独立优化的错误叶型生成次数
	
	% 从Initial.txt读取边界数据
	fid = fopen('result\\Initial.txt', 'r');
	data = textscan(fid, '%f %f', D); % 读取D行，每行两个浮点数
	fclose(fid);
	% 提取Lb和Ub
	Lb = [data{1}]'; % 转换为行向量
	Ub = [data{2}]'; 
	% 确保边界维度正确
	assert(size(Lb,2) == D, 'Lb维度错误');
	assert(size(Ub,2) == D, 'Ub维度错误');
    
    for run_time=1:run_Maxtime
        %% Initialization
		New_Lb = Lb;
		New_Ub = Ub;
		Boundary = [New_Ub; New_Lb]; % 保持原格式[上界; 下界]
        error_iii = 0;

        %Initialize the training archive
        count_Lb_extension=0;
        count_Ub_extension=0;
		Alhs = lhsdesign(Nt,D); % 拉丁超立方体采样
        Arc = New_Lb + Alhs.* (New_Ub-New_Lb);
        ii = 0;
		
        fprintf('%s%s\n', 'count_cal---','---tploss---');
        while (ii<size(Arc,1))
            ii=ii+1;
            fprintf('    %d ', ii);%显示目标函数被调用次数
            POP_I=Arc(ii,1:D);
            fid1=fopen('geom_ctl.dat','w+'); %每次更新几何控制参数，则每次都要重新打开geom_ctl.dat文件，用完后再重新关闭文件，这样保证每次文件中只有一组数据。
            fprintf(fid1,'%f \n',POP_I);	
            fclose(fid1);
            [obj,recal_index] = fitness(POP_I); %初始化的目标函数存入Arc，调用目标函数
           %*****  排除错误解，重新采样，不让错误解进入样本库  ****%
            while (obj==1 || obj<=0.020) 
                POP_I = New_Lb + (New_Ub -New_Lb) .* rand(1, D);%生成随机样本
                fid1=fopen('geom_ctl.dat','w+'); %每次更新几何控制参数，则每次都要重新打开geom_ctl.dat文件，用完后再重新关闭文件，这样保证每次文件中只有一组数据。
                fprintf(fid1,'%f \n',POP_I);	
                fclose(fid1);
                [obj,recal_index] = fitness(POP_I);
                Arc(ii,1:D)=POP_I;  %要把新变量传递给POP0矩阵，从而传递给Data
            end
           %************   over   ***********
           Arc(ii,D+1)=obj;%将目标函数赋给Arc的第D+1列
           % --------将最优叶型保存到输出文件中--------------
			nowobj = obj;
			if nowbest > nowobj
				sourceFile = 'iblade2D.dat';
				destinationFolder = 'result';
				destinationFile = fullfile(destinationFolder, sprintf('iblade2D_%d.dat', run_time));
				% 复制文件并重命名
				copyfile(sourceFile, destinationFile);
			end
		    nowbest = min(Arc(:,end));
        end
        FEnum = size(Arc,1);
        %--------initialize over--------------
        fprintf('%s\n', '*************************initialize over*************************');
        %Optimization
        while FEnum < FEmax
			%确保是经过优化后的上下边界
			New_Ub = Boundary(1, :);
			New_Lb = Boundary(2, :);
			
            ArcDec = Arc(:,1:D);
            ArcObj = Arc(:,end);
            %Train a GP model
            GPmodel = dacefit(ArcDec,ArcObj,'regpoly0','corrgauss',theta,1e-5.*ones(1,D),100.*ones(1,D));
            %Initialize a parent population
            Plhs = lhsdesign(Np,D);
            Parent = New_Lb + Plhs.* (New_Ub-New_Lb);
            g = 0;
            %Generate an offspring
            while g < gmax
                g = g+1;
                Offspring = GA(Parent,Boundary); % 使用遗传算法生成新的后代种群
                Popcom = [Parent;Offspring]; % 将父代和后代合并成一个混合种群 Popcom
                N = size(Popcom,1);
                for i = 1: N
                    [PopObj(i,1),~,MSE(i,1)] = predictor(Popcom(i,:),GPmodel); % % 使用代理模型来评估每个个体的适应度值 PopObj(i,1) 和均方误差 MSE(i,1)
                end
                [~,index] = sort(PopObj,'ascend'); % 按照适应度值 PopObj 从小到大排序，并获取排序后的索引 index
                Parent = Popcom(index(1:Np),:); % 根据排序结果选择前 Np 个最优个体作为新的父代种群 Parent
                MSEp = MSE(index(1:Np),:); % 更新父代种群的均方误差 MSEp
                ParObj = PopObj(index(1:Np)); % 更新父代种群的适应度值 ParObj
            end
            %Infill criterion based on EI
            Abest = min(Arc(:,end)); % 从历史存档 Arc 中获取当前最优适应度值 Abest
            s = sqrt(MSEp);
            lamda = (repmat(Abest,Np,1)-ParObj)./s;
            % 对父代种群中的每个个体进行期望改进（EI）计算
            for i = 1:Np
                EI(i,1) = (Abest-ParObj(i)).* Gaussian_CDF(lamda(i)) + s(i)*Gaussian_PDF(lamda(i)); 
            end
            [~,index] = max(EI); % 找到具有最大期望改进值 EI 的个体索引 index
            Popreal = Parent(index,:);
            %--------开始计算目标函数-------------
            FEnum = FEnum + 1;
            fprintf('%s %d %s\n','----------------count_cal:',FEnum,'----------------');%显示目标函数被调用次数
            fid1=fopen('geom_ctl.dat','w+'); %每次更新几何控制参数，则每次都要重新打开geom_ctl.dat文件，用完后再重新关闭文件，这样保证每次文件中只有一组数据。
            fprintf(fid1,'%f \n',Popreal);	
            fclose(fid1);
            [obj,recal_index] = fitness(Popreal); %初始化的目标函数存入Arc，调用目标函数
           %*****  排除错误解，重新采样，不让错误解进入样本库  ****%  更改删除原种群中EI最大的样本点，选取第二大EI值的样本点进行实际评估 bin
            iii = 0;
            while (obj==1 || obj<=0.020)
                iii = iii + 1;
                error_iii = error_iii + 1;
                FEnum = FEnum + 1;
                if iii <10
                    EI(index) = -Inf; % 将原EI最大值设置为负无穷，这样它不会再被选中为最大值
                    [~ , index] = max(EI); % 找到第二大期望改进值 EI 的个体索引 second_index
                    Popreal = Parent(index, :); % 选择第二大期望改进值对应的个体
                    fid1=fopen('geom_ctl.dat','w+'); %每次更新几何控制参数，则每次都要重新打开geom_ctl.dat文件，用完后再重新关闭文件，这样保证每次文件中只有一组数据。
                    fprintf(fid1,'%f \n',Popreal);	
                    fclose(fid1);
                    [obj,recal_index] = fitness(Popreal);
                else 
                    Popreal = New_Lb + (New_Ub -New_Lb) .* rand(1, D);%生成随机样本
                    fid1=fopen('geom_ctl.dat','w+'); %每次更新几何控制参数，则每次都要重新打开geom_ctl.dat文件，用完后再重新关闭文件，这样保证每次文件中只有一组数据。
                    fprintf(fid1,'%f \n',Popreal);	
                    fclose(fid1);
                    [obj,recal_index] = fitness(Popreal);
                end
            end
            %************   over   ***********    
            Popreal(:,D+1) = obj;
            %Update the training archive
            Arc = [Arc; Popreal]; %#ok<*AGROW>   
        
            %*****计算此时最优解的代理模型精度******
            [Pred_best,~,~]=predictor(Popreal(:,1:D),GPmodel);
            eta=abs(Pred_best-obj)/obj;  %相对误差
            fprintf('eta at current best solution: %.2f%%\n', eta*100);
            % eta_history(FEnum-Nt,:) = [FEnum, eta];
            % --------将最优叶型保存到输出文件中--------------
            nowobj = obj;
            if nowbest > nowobj
                sourceFile = 'iblade2D.dat';
                destinationFolder = 'result';
                destinationFile = fullfile(destinationFolder, sprintf('iblade2D_%d.dat', run_time));
                % 复制文件并重命名
                copyfile(sourceFile, destinationFile);
            end
            nowbest = min(Arc(:,end));

            % %      开始准备自适应调节变量范围adaptive_range
            % %----------------------------------------------
            % %----  扩展上下边界  -------
            % POP_II=Popreal(:,1:D);
            % Lb_index = [];  % 初始化上下边界存储序列号的向量
            % Ub_index = [];
            % %最优解预测精度
            % if obj <= (min(Arc(:,end)) * 1.01)   %小于或者接近当前历史最优值
            %     for i = 1:D % bin
            %         if Popreal(1, i) <= (New_Lb(1, i) + 2e-1) % 检查Popreal是否达到了下边界
            %             % 达到边界，扩展边界 
            %             New_Lb(1, i) = New_Lb(1, i) - abs(New_Ub(1,i)-New_Lb(1,i)) * extension_ratio;
            %             count_Lb_extension=count_Lb_extension+1; %当此时存在多个元素到达边界时则会产生多次计数
            %             POP_II(1,i)=POP_II(1,i)-abs(New_Ub(1,i)-New_Lb(1,i)) * extension_ratio;  %对该设计变量本身进行扩展
            %             % 将满足条件的点的序列号存储在 Lb_index 向量中
            %             Lb_index = [Lb_index, i];
            %         end
            %         if Popreal(1, i) >= (New_Ub(1, i) - 2e-1) % 检查Popreal是否达到了上边界
            %             % 达到边界，扩展边界
            %             New_Ub(1, i) = New_Ub(1, i) +abs(New_Ub(1,i)-New_Lb(1,i)) * extension_ratio;
            %             count_Ub_extension=count_Ub_extension+1;  %当此时存在多个元素到达边界时则会产生多次计数
            %             POP_II(1,i)=POP_II(1,i)+abs(New_Ub(1,i)-New_Lb(1,i)) * extension_ratio;  %对该设计变量本身进行扩展
            %             % 将满足条件的点的序列号存储在 Ub_index 向量中
            %             Ub_index = [Ub_index, i];             
            %         end
            %     end
            %     fprintf('Lb_index: %.0f ', Lb_index); % 打印Lb_index所有值，自动去小数
            %     fprintf('\n'); % 换行
            %     fprintf('Ub_index: %.0f ', Ub_index); % 打印Ub_index所有值
            %     fprintf('\n'); % 换行
            % end            
            % fprintf('%s %d  ','count_Lb_extension:',count_Lb_extension);%显示下边界被扩展的次数
            % fprintf('%s %d\n','count_Ub_extension:',count_Ub_extension);%显示上边界被扩展的次数
            % 
            % %----------生成扩展后新设计变量，并计算真实值，存入样本库----------
            % if ~isequal(POP_II,Popreal(:,1:D))  %如果存在边界扩展，则要对新设计变量进行计算
            %     FEnum = FEnum + 1;
            %     fprintf('%s %d %s\n','----------------count_cal:',FEnum,'----------------');%显示目标函数被调用次数
            %     extension_begin=Popreal; %拓展前，先把之前的最优解Popreal赋给extension_begin
            %     extension_begin_Num = size(Arc,1); %先把此时的序列号存储好
            %     fid1=fopen('geom_ctl.dat','w+'); %每次更新几何控制参数，则每次都要重新打开geom_ctl.dat文件，用完后再重新关闭文件，这样保证每次文件中只有一组数据。
            %     fprintf(fid1,'%f \n',POP_II);	
            %     fclose(fid1);
            %     [obj_new,recal_index] = fitness(POP_II);
            % 
            %     %*****  排除无效拓展，重新采样，不让无效拓展进入样本库  ****% 
            %     iii = 0;
            %     while ((obj_new==1) || (obj_new>obj * 1.01) || (obj_new<=0.020)) && iii<10 % 往回收缩边界
            %         iii = iii+1;
            %         error_iii = error_iii + 1;
            %         FEnum = FEnum + 1;
            %         for i = 1:length(Lb_index)
            %             New_Lb(1, Lb_index(i)) = New_Lb(1, Lb_index(i)) + abs(New_Ub(1,Lb_index(i))-New_Lb(1,Lb_index(i))) * extension_ratio * 0.25;
            %             POP_II(1, Lb_index(i)) = POP_II(1, Lb_index(i)) + abs(New_Ub(1, Lb_index(i)) - New_Lb(1, Lb_index(i))) * extension_ratio * 0.25;
            %         end
            %         for i = 1:length(Ub_index)
            %             New_Ub(1, Ub_index(i)) = New_Ub(1, Ub_index(i)) - abs(New_Ub(1,Ub_index(i))-New_Lb(1,Ub_index(i))) * extension_ratio * 0.25;
            %             POP_II(1, Ub_index(i)) = POP_II(1, Ub_index(i)) - abs(New_Ub(1, Ub_index(i)) - New_Lb(1, Ub_index(i))) * extension_ratio * 0.25; 
            %         end
            %         fid1=fopen('geom_ctl.dat','w+'); %每次更新几何控制参数，则每次都要重新打开geom_ctl.dat文件，用完后再重新关闭文件，这样保证每次文件中只有一组数据。
            %         fprintf(fid1,'%f \n',POP_II);	
            %         fclose(fid1);
            %         [obj_new,recal_index] = fitness(POP_II);
            %     end
            % 
            %     Popreal(:,1:D)=POP_II;%将扩展的POP_II传递给Popreal，从而传递Arc
            %     Popreal(:,D+1) = obj_new;
            %     Arc = [Arc; Popreal];
            % 
            %    % -----反复拓展边界，直到性能不再提升-----
            %     while obj_new < obj && obj_new>0.020
			% 		% --------将最优叶型保存到输出文件中--------------
            %         nowobj = obj_new;
			% 		if nowbest > nowobj
			% 			Boundary = [New_Ub;New_Lb];%更新上下边界
			% 			sourceFile = 'iblade2D.dat';
			% 			destinationFolder = 'result';
			% 			destinationFile = fullfile(destinationFolder, sprintf('iblade2D_%d.dat', run_time));
			% 			% 复制文件并重命名
			% 			copyfile(sourceFile, destinationFile);
			% 		end
			% 		nowbest = min(Arc(:,end));
            % 
            %         obj = obj_new; % 更新最佳性能, 如果上一步计算失败，重新采样，此时的扩展边界就失去了意义 bin
            %         FEnum = FEnum + 1;
            %         fprintf('%s %d %s\n','----------------count_cal:',FEnum,'----------------');%显示目标函数被调用次数
            %         % 扩展边界
            %         for i = 1:length(Lb_index)
            %             New_Lb(1, Lb_index(i)) = New_Lb(1, Lb_index(i)) -abs(New_Ub(1, Lb_index(i))-New_Lb(1, Lb_index(i))) *extension_ratio;
            %             count_Lb_extension=count_Lb_extension+1; %当此时存在多个元素到达边界时则会产生多次计数
            %             POP_II(1, Lb_index(i)) = POP_II(1, Lb_index(i)) - abs(New_Ub(1, Lb_index(i)) - New_Lb(1, Lb_index(i))) * extension_ratio;
            %         end
            %         for i = 1:length(Ub_index)
            %             New_Ub(1, Ub_index(i)) = New_Ub(1, Ub_index(i)) +abs(New_Ub(1, Ub_index(i))-New_Lb(1, Ub_index(i))) *extension_ratio;
            %             count_Ub_extension=count_Ub_extension+1; %当此时存在多个元素到达边界时则会产生多次计数
            %             POP_II(1, Ub_index(i)) = POP_II(1, Ub_index(i)) + abs(New_Ub(1, Ub_index(i)) - New_Lb(1, Ub_index(i))) * extension_ratio;
            %         end
            %         fprintf('%s %d  ','count_Lb_extension:',count_Lb_extension);%显示下边界被扩展的次数
            %         fprintf('%s %d\n','count_Ub_extension:',count_Ub_extension);%显示上边界被扩展的次数
            %         % 计算新的真实函数值
            %         fid1 = fopen('geom_ctl.dat', 'w+');
            %         fprintf(fid1, '%f \n', POP_II);    
            %         fclose(fid1);
            %         [obj_new, recal_index] = fitness(POP_II);
            % 
            %         %*****  排除无效拓展，重新采样，不让无效拓展进入样本库  ****% 
            %         iii = 0;
            %         while ((obj_new==1) || (obj_new>obj * 1.01) || (obj_new<=0.020)) && iii<10 % 往回收缩边界
            %             iii = iii+1;
            %             error_iii = error_iii + 1;
            %             FEnum = FEnum + 1;
            %             for i = 1:length(Lb_index)
            %                 New_Lb(1, Lb_index(i)) = New_Lb(1, Lb_index(i)) + abs(New_Ub(1,Lb_index(i))-New_Lb(1,Lb_index(i))) * extension_ratio * 0.25;
            %                 POP_II(1, Lb_index(i)) = POP_II(1, Lb_index(i)) + abs(New_Ub(1, Lb_index(i)) - New_Lb(1, Lb_index(i))) * extension_ratio * 0.25;
            %             end
            %             for i = 1:length(Ub_index)
            %                 New_Ub(1, Ub_index(i)) = New_Ub(1, Ub_index(i)) - abs(New_Ub(1,Ub_index(i))-New_Lb(1,Ub_index(i))) * extension_ratio * 0.25;
            %                 POP_II(1, Ub_index(i)) = POP_II(1, Ub_index(i)) - abs(New_Ub(1, Ub_index(i)) - New_Lb(1, Ub_index(i))) * extension_ratio * 0.25; 
            %             end
            %             fid1=fopen('geom_ctl.dat','w+'); %每次更新几何控制参数，则每次都要重新打开geom_ctl.dat文件，用完后再重新关闭文件，这样保证每次文件中只有一组数据。
            %             fprintf(fid1,'%f \n',POP_II);	
            %             fclose(fid1);
            %             [obj_new,~] = fitness(POP_II);
            %         end
            % 
            %         Popreal(:,1:D)=POP_II;%将重新生成的POP_II传递给Popreal，从而传递Arc
            %         Popreal(:,D+1) = obj_new;
            %         Arc = [Arc; Popreal];
            %     end
            %     extension_end_Num = size(Arc,1);
            %     %************   over   ***********			
            %     filename=sprintf('result\\extension_run_%d.txt',run_time);
            %     fid2=fopen(filename,'a+');
            %     fprintf(fid2,'%s','Lb_index = ');
            %     fprintf(fid2,'%d ',Lb_index);
            %     fprintf(fid2,'\n');
            %     fprintf(fid2,'%s','Ub_index = ');
            %     fprintf(fid2,'%d ',Ub_index);
            %     fprintf(fid2,'\n');
            %     fprintf(fid2,'eta at current best solution: %f\n', eta);        
            %     fprintf(fid2,'%s  %d        %s  %d\n','count_Lb_extension:',count_Lb_extension,'count_Ub_extension:',count_Ub_extension);%上下边界扩展了几次；
            %     fprintf(fid2,'%s %d   %s %d\n','extension_begin_Num:',extension_begin_Num, 'extension_end_Num:',extension_end_Num); %第几次开始自适应触发
            %     fprintf(fid2,'%f ',extension_begin);%触发自适应的那个解
            %     fprintf(fid2,'\n');
            %     fprintf(fid2,'%f ',Popreal);%自适应后的那个解
            %     fprintf(fid2,'\n\n');
            %     fclose(fid2);
            % end
            % %-----------------adaptive_range over----------------------------
        end

        %--------将结果输出到文件中--------------
        filename1=sprintf('result\\best_result_run_%d.txt',run_time);
        filename2=sprintf('result\\Total_result_run_%d.txt',run_time);
        
        [Pbest,I] = min(Arc(:,end));
        fprintf('%s %d\n','Global_best:',Pbest);
        gbest=Arc(I,:);
        save(filename1,'gbest','-ascii');
        save(filename2,'Arc','-ascii');


        Global_best(1,run_time)=Pbest;%将每一整局独立优化的最优值存储在Global_best中，run_time值为x整局
        count_Lb_extension_run(1,run_time) = count_Lb_extension;
        count_Ub_extension_run(1,run_time) = count_Ub_extension;
        %-----OutPut convergency history------------
        % i=0;
        % filename3=sprintf('result\\Total_eta_run_%d.txt',run_time);
        filename4=sprintf('result\\history_run_%d.plt',run_time);
        % fid3=fopen(filename3,'w');
        fid4=fopen(filename4,'w');
        % while(i<(FEnum-Nt))
        %     i=i+1;
        %     fprintf(fid3,'%d    %f \n',eta_history(i,1),eta_history(i,2));
        % end
        j=0;
        while(j<size(Arc,1))
            j=j+1;
            [A,~]=min(Arc(1:j,end));
            fprintf(fid4,'%d    %f \n',j,A);
        end
        % fclose(fid3);
        fclose(fid4);
        destinationFolder = 'result';
        destinationFile = fullfile(destinationFolder, sprintf('final_%d.txt', run_time));
        fid5=fopen(destinationFile,'w');
        for i=1:D
            New_Lb(:,i)=min(Arc(:,i));
            New_Ub(:,i)=max(Arc(:,i));
            fprintf(fid5,'%f   %f \n',New_Lb(:,i),New_Ub(:,i));
        end
        fclose(fid5);

        %保存本次优化错误叶型生成次数
        error_s(run_time) = error_iii;
    end

    %%计算统计量
    %---最优值统计---
    mean_Global_best = mean(Global_best);
    std_Global_best = std(Global_best);
    max_Global_best = max(Global_best);
    min_Global_best = min(Global_best);
    median_Global_best = median(Global_best);
    %---自适应参量统计-----
    mean_count_Lb_extension = mean(count_Lb_extension_run);
    std_count_Lb_extension = std(count_Lb_extension_run);
    max_count_Lb_extension = max(count_Lb_extension_run);
    min_count_Lb_extension = min(count_Lb_extension_run);
    median_count_Lb_extension = median(count_Lb_extension_run);
    mean_count_Ub_extension = mean(count_Ub_extension_run);
    std_count_Ub_extension = std(count_Ub_extension_run);
    max_count_Ub_extension = max(count_Ub_extension_run);
    min_count_Ub_extension = min(count_Ub_extension_run);
    median_count_Ub_extension = median(count_Ub_extension_run);
    
    fid_total=fopen('result\\total_statistic.txt','w');
    fprintf(fid_total,'%s  %f\n','mean_Global_best: ',mean_Global_best);
    fprintf(fid_total,'%s  %f\n','std_Global_best: ',std_Global_best);
    fprintf(fid_total,'%s  %f\n','min_Global_best: ',min_Global_best);
    fprintf(fid_total,'%s  %f\n','max_Global_best: ',max_Global_best);
    fprintf(fid_total,'%s  %f\n','median_Global_best: ',median_Global_best);
    fprintf(fid_total,'%s  %f\n','mean_count_Lb_extension: ',mean_count_Lb_extension);
    fprintf(fid_total,'%s  %f\n','std_count_Lb_extension: ',std_count_Lb_extension);
    fprintf(fid_total,'%s  %f\n','max_count_Lb_extension: ',max_count_Lb_extension);
    fprintf(fid_total,'%s  %f\n','min_count_Lb_extension: ',min_count_Lb_extension);
    fprintf(fid_total,'%s  %f\n','median_count_Lb_extension: ',median_count_Lb_extension);
    fprintf(fid_total,'%s  %f\n','mean_count_Ub_extension: ',mean_count_Ub_extension);
    fprintf(fid_total,'%s  %f\n','std_count_Ub_extension: ',std_count_Ub_extension);
    fprintf(fid_total,'%s  %f\n','max_count_Ub_extension: ',max_count_Ub_extension);
    fprintf(fid_total,'%s  %f\n','min_count_Ub_extension: ',min_count_Ub_extension);
    fprintf(fid_total,'%s  %f\n','median_count_Ub_extension: ',median_count_Ub_extension);
    fclose(fid_total);
    toc

    %存储错误叶型生成数据
    result_dir = fullfile(pwd, 'result');
    output_file = fullfile(result_dir, 'error statistics.txt');
    fid6 = fopen(output_file, 'w');          % 以写入模式打开文件
    if fid6 == -1
        error('文件创建失败，请检查权限！');  % 错误处理
    end
    % 格式化写入数据
    data_mean = mean(error_s);
    for k = 1:numel(error_s)
        fprintf(fid6, 'run_time%d: %d\n', k, error_s(k));
    end
    data_ratio = mean(error_s) / FEmax * 100;
    fprintf(fid6, '\nAverage_error_statistics: %.2f', data_mean);
    fprintf(fid6, '\nRatio_error_statistics: %.2f%%', data_ratio);
    fclose(fid6);

return