%function [lfp_rws,lfp_rws_pre]=run165(n_stimuli,sequential_delay)
n_stimuli=1;
sequential_delay=0;
%root_dir = ['D:\zsk\WM paper\code and parameters_zjpzyw\newrecord\record',num2str(n_stimuli),'_',num2str(sequential_delay),'\record'];
root_dir = [''];
 if ~exist(root_dir,'dir')
        mkdir (root_dir)
    end
idx = 1;

datanum=1;
%n_stimuli=1;
pyr_input=[500,700];
t_end=2000;
t_stimuli='u';
mini_diff=0;



%% Part I: simulation parameters:

% number of pyramidal cells and interneurons

    pyr_Nneurons_s = 4096; %soma/initial axonal
    pyr_Nneurons_d1 = 4096; %proximal dendrite
    
    inh_Nneurons_pv = 512; %parvalbumin %no cr
    inh_Nneurons_cb = 512; %calbindin %no cr

    dt = 0.02;

% Transmission delays:
    pyr_s_trans_delays = zeros(pyr_Nneurons_s,1);
    pyr_d1_trans_delays = zeros(pyr_Nneurons_d1,1);

    inh_pv_trans_delays = zeros(inh_Nneurons_pv,1);
    inh_cb_trans_delays = zeros(inh_Nneurons_cb,1);

%%%%%%
% *********************
%  Membrane Parameters
% *********************

% ** Pyramidal Cells **
    pyr_Cm = 0.5; % [nF]
    pyr_g_leak = 0.025; % [microSiemens] %beta 0.025
    pyr_g_leak_d1 = 0.025; % [microSiemens] %beta 0.025
    pyr_Vm_leak = -70; % [mV]
    pyr_Vm_thresh = -50; % [mV]
    pyr_Vm_reset = -60; % [mV]
    pyr_tau_ref = 2; % [ms]

% ** Interneurons **
    inh_Cm_pv = 0.2; % [nF]
    inh_Cm_cb = 0.8; % 10/19 2018
    inh_g_leak_cb = 0.016; % [microS] beta 0.020
    inh_g_leak_pv = 0.020; % beta 0.020
    inh_Vm_leak_cb = -76; % [mV] -70
    inh_Vm_leak_pv = -86; % [mV] beta -70
    inh_Vm_thresh = -50; % [mV]
    inh_Vm_reset = -60; % [mV]
    inh_tau_ref = 1; % [ms]

%%%%%%
% *********************
%  Synapese parameters
% *********************

% ** AMPA **
    syn_Ve_AMPA = 0; % [mV]
    syn_tau_s_AMPA = 2; % [ms]
    syn_alpha_s_AMPA = 1;

% ** NMDA **
    syn_Ve_NMDA = 0; % [mV]
    syn_Mg_concentration = 1; % [mM]
    syn_tau_x_NMDA = 2; % [ms]
    syn_alpha_x_NMDA = 1; %
    syn_tau_s_NMDA = 100; % [ms]
    syn_alpha_s_NMDA = 0.5; % [1/ms]
    
    syn_tau_f_NMDA = 50; % [ms] stp_f 2018/10/12
%     syn_alpha_u_NMDA = 1; %stp 2018/10/12
    
    syn_U_NMDA = 0.4; %stp 10/19
%     K_stp = 2500; %stp 2018/10/12

% ** GABA **
    syn_Ve_GABA = -70; % [mV]
    syn_tau_s_GABA = 10; % [ms]
    syn_alpha_s_GABA = 1/3; % [ms]

%%%%%%
% ****************
%  External 
% ****************

% External Poisson spiking train (noise)
    pyr_ext_noise_rate = 1.0; % [1/ms];
    %equivalently: 1000 synapses with 1.0Hz input;
    inh_ext_noise_rate = 1.0;
    %equivalently: 1000 synapses with 1.0Hz input;


%%%%%%    
% **********************
%  Maximal Conductances,
%  Connectivity Coefficients
% ***************************

% External (noise) to pyramidal cells and interneurons:
    pyr_ext_noise_g_AMPA = 0.00390;%0.00390; % [microS] beta 0.00248
    inh_pv_ext_noise_g_AMPA = 0.0024;%0.0024%0.0019; % [microS] 0.0019
    inh_cb_ext_noise_g_AMPA = 0.0029;%0.0029%0.0022; % [microS] 0.0019

% Connectivity

    gc1 = 1.2; %beta 0.75 d1 0.9 2018/9/26 1.2
    gc2 = 0.25; 
    p1 = 0.6; % 0.5
    p2 = 0.4; %beta 0.3 d1
    
    %sequential_delay = 0; % simultaneously: 0; sequentially: >0
    sequential_tensify = [1.5 1.5];
    
    sigma_pyr_s2pyr_s = 12.76;%1 * 2^(1/2); %narrow:9.25 %wide:11.25 %new model:1.1
    sigma_pyr_s2inh_pv = 5 * 2^(1/2); %new21
    sigma_pyr_s2inh_cb = 1 * 2^(1/2); %new model:no construction
    
    sigma_inh_pv2pyr_s = 6 * 2^(1/2); %new21
    sigma_inh_pv2inh_cb = 1 * 2^(1/2); 
    sigma_inh_pv2inh_pv = 0; 
    
    sigma_inh_cb2pyr_d1 = 0.8 * 2 * pi; %new model
    sigma_inh_cb2inh_pv = 0;

    J_plus_pyr_s2pyr_s = 6.5; %narrow:2.72 %wide:3.62 %new model:17.5
    J_plus_pyr_s2inh_pv = 35.0; %new21
    J_plus_pyr_s2inh_cb = 70.0; 

    %J_plus_inh2pyr = 3.62;
    
    popA_G_pyr_s2pyr_s_AMPA = 0.0000750*1.5;%zhou%0.393nS 0.251(compte.)
    popA_G_pyr_s2inh_pv_AMPA = 0.0001600;%zhou%0.304nS 0.192(compte.)
    popA_G_pyr_s2inh_cb_AMPA = 0.0002400;
    
    popA_G_pyr_s2pyr_s_NMDA = 0.0000750;%0.214nS 0.274nS(compte.)%0.0001905
    popA_G_pyr_s2inh_pv_NMDA = 0.0000400*1.5;%0.164nS 0.212nS(compte.)%0.0001460
    popA_G_pyr_s2inh_cb_NMDA = 0.0002400;

    popA_G_inh_pv2pyr_s_GABA = 0.000760*0.3*1.5;
    popA_G_inh_cb2pyr_d1_GABA = 0.0006070*3;
    
    popA_G_inh_cb2inh_pv_GABA = 0.0000300*3;
    popA_G_inh_pv2inh_cb_GABA = 0.035000*0.3;
    popA_G_inh_pv2inh_pv_GABA = 0.000200*0.3;
    
    g_Ca_s = 0.0015000;
    g_Ca_cb = 0.0010000;

%%%%%%%
% ******************************
%  Input stimuli - 'free style'
% ******************************


    if n_stimuli <=0
        error ('Please use a postive number for the number of stimuli in the cue');
    end

    if t_stimuli == 'u'
        theta_stim = 180/n_stimuli:360/n_stimuli:360-180/n_stimuli;
    elseif t_stimuli == 'r'
        theta_stim = get_rand_stimuli(n_stimuli,mini_diff);
    elseif t_stimuli == 'c'
        theta_stim = mini_diff;
    else
        error ('Please use the correct type of stimuli within "u" and "r"');
    end

    fname_input = [root_dir,'input_',num2str(datanum),'.txt'];
    f_input = fopen(fname_input, 'w');
    fprintf(f_input, '%f,', theta_stim);
    fclose(f_input);

    sigma_stim = 2;%2;
    max_stim_current = 0.5; %beta 0.4
    
    
    
% start point and end point of the input
    % pyr_input = [250, 500] ;

%%%%%%%%
% ****************
%  Initial Values
% ****************

    pyr_popA_init_Vm = -51;% [mV]
    pyr_popA_init_x_NMDA = 0.1;  %
    pyr_popA_init_s_NMDA = 0.05 ;
    pyr_popA_init_s_AMPA = 0.05 ;
    
    pyr_popA_init_U_NMDA = 0.2; %stp 2018/10/12

    inh_popA_init_Vm = -51;% [mV]
    inh_popA_init_s_GABA = 0.5; %

    pyr_popA_init_s_AMPA_ext_noise = 0;%.6;
    inh_popA_init_s_AMPA_ext_noise = 0;%.6;

    % disp('All inputs are valid! & Let the simulation begin :)');

%% Random number seed
    rand_seed = sum(100*clock);
    
% old version of random seed generator
%    rand('state',rand_seed);

% new version of randoom seed generator
    rng (rand_seed);

%% Initial Network for simulations

% For transmission delay:
    pyr_s_popA_transmission_delay_t_steps = max(round(pyr_s_trans_delays/dt),1);
    pyr_s_popA_delay_bank_size = max(pyr_s_popA_transmission_delay_t_steps);
    pyr_s_popA_WhoFired_bank = zeros(pyr_Nneurons_s,pyr_s_popA_delay_bank_size);
    pyr_s_popA_ind_now_in_bank = 0;
    
    pyr_d1_popA_transmission_delay_t_steps = max(round(pyr_d1_trans_delays/dt),1);
    pyr_d1_popA_delay_bank_size = max(pyr_d1_popA_transmission_delay_t_steps);
    pyr_d1_popA_WhoFired_bank = zeros(pyr_Nneurons_d1,pyr_d1_popA_delay_bank_size);
    pyr_d1_popA_ind_now_in_bank = 0;

% same for the inh population:
    inh_pv_popA_transmission_delay_t_steps = max(round(inh_pv_trans_delays/dt),1);
    inh_pv_popA_delay_bank_size = max(inh_pv_popA_transmission_delay_t_steps);
    inh_pv_popA_WhoFired_bank = zeros(inh_Nneurons_pv,inh_pv_popA_delay_bank_size);
    inh_pv_popA_ind_now_in_bank = 0;

    inh_cb_popA_transmission_delay_t_steps = max(round(inh_cb_trans_delays/dt),1);
    inh_cb_popA_delay_bank_size = max(inh_cb_popA_transmission_delay_t_steps);
    inh_cb_popA_WhoFired_bank = zeros(inh_Nneurons_cb,inh_cb_popA_delay_bank_size);
    inh_cb_popA_ind_now_in_bank = 0;

% Initial values:
    load('pyr_s_x_NMDA_init.mat');
    load('pyr_s_s_NMDA_init.mat');
    load('pyr_s_s_AMPA_init.mat');
    pyr_s_x_NMDA = pyr_s_x_NMDA_init;
    pyr_s_s_NMDA = pyr_s_s_NMDA_init;
    pyr_s_s_AMPA = pyr_s_s_AMPA_init;
    pyr_s_u_NMDA = pyr_popA_init_U_NMDA(ones(pyr_Nneurons_s,1)); %stp  2018/10/12
    
    load('pyr_d1_x_NMDA_init.mat');
    load('pyr_d1_s_NMDA_init.mat');
    load('pyr_d1_s_AMPA_init.mat');
    pyr_d1_x_NMDA = pyr_d1_x_NMDA_init;
    pyr_d1_s_NMDA = pyr_d1_s_NMDA_init;
    pyr_d1_s_AMPA = pyr_d1_s_AMPA_init;
    
    load('inh_pv_s_GABA_init.mat');
    load('inh_cb_s_GABA_init.mat');
    inh_pv_s_GABA = inh_pv_s_GABA_init;
    inh_cb_s_GABA = inh_cb_s_GABA_init;

    load('pyr_s_Vm_prev_init.mat');
    load('pyr_s_Vm_new_init.mat');
    load('pyr_s_ext_noise_s_AMPA_init.mat');
    pyr_s_Vm_prev = pyr_s_Vm_prev_init;
    pyr_s_Vm_new = pyr_s_Vm_new_init;
    pyr_s_ext_noise_s_AMPA = pyr_s_ext_noise_s_AMPA_init;
    
    load('pyr_d1_Vm_prev_init.mat');
    load('pyr_d1_Vm_new_init.mat');
    load('pyr_d1_ext_noise_s_AMPA_init.mat');
    pyr_d1_Vm_prev = pyr_d1_Vm_prev_init;
    pyr_d1_Vm_new = pyr_d1_Vm_new_init;
    pyr_d1_ext_noise_s_AMPA = pyr_d1_ext_noise_s_AMPA_init;
    
    load('inh_pv_Vm_prev_init.mat');
    load('inh_pv_Vm_new_init.mat');
    load('inh_pv_ext_noise_s_AMPA_init.mat');
    inh_pv_Vm_prev = inh_pv_Vm_prev_init;
    inh_pv_Vm_new = inh_pv_Vm_new_init;
    inh_pv_ext_noise_s_AMPA = inh_pv_ext_noise_s_AMPA_init;

    load('inh_cb_Vm_prev_init.mat');
    load('inh_cb_Vm_new_init.mat');
    load('inh_cb_ext_noise_s_AMPA_init.mat');
    inh_cb_Vm_prev = inh_cb_Vm_prev_init;
    inh_cb_Vm_new = inh_cb_Vm_new_init;
    inh_cb_ext_noise_s_AMPA = inh_cb_ext_noise_s_AMPA_init;

    pyr_s_WhoFired = [];
    pyr_s_SpikeTimes = []; %#ok<NASGU> % ignore this warning
    load('pyr_s_LastTimeEachFired_init.mat');
    pyr_s_LastTimeEachFired = pyr_s_LastTimeEachFired_init - 250; 
    % the last spike of each neuron.
    
    pyr_d1_WhoFired = [];
    pyr_d1_SpikeTimes = []; 
    load('pyr_d1_LastTimeEachFired_init.mat');
    pyr_d1_LastTimeEachFired = pyr_d1_LastTimeEachFired_init - 250; 
    
    inh_pv_WhoFired = [];
    inh_pv_SpikeTimes = []; 
    load('inh_pv_LastTimeEachFired_init.mat');
    inh_pv_LastTimeEachFired = inh_pv_LastTimeEachFired_init - 250; 
    
    inh_cb_WhoFired = [];
    inh_cb_SpikeTimes = []; 
    load('inh_cb_LastTimeEachFired_init.mat');
    inh_cb_LastTimeEachFired = inh_cb_LastTimeEachFired_init - 250; 
    
    % local field potential
    lfp = zeros(t_end/dt+1,1); %zhou lfp
    lfp_rws = zeros(t_end/dt+1,1); %zhou lfp
    lfp_ieee = zeros(t_end/dt+1,1); %zhou lfp
    
    lfp_rws_pre = zeros(t_end/dt+1,1); %pre
    lfp_rws_nonpre = zeros(t_end/dt+1,1); %pre
    
    %firing rate
    fr=zeros(2001,4096);
    fr_1=zeros(t_end/dt+1,4096);
    fr1=zeros(2001,512);
    FS_prefer_total_I=zeros(t_end/dt+1,1);
    FS_nonprefer_total_I=zeros(t_end/dt+1,1);
    NFS_prefer_total_I=zeros(t_end/dt+1,1);
    NFS_nonprefer_total_I=zeros(t_end/dt+1,1);
   
    % record the Vm of each neuron
     pyr_s_Vm = zeros(t_end/dt+1,pyr_Nneurons_s); %zhou record
     pyr_d1_Vm = zeros(t_end/dt+1,pyr_Nneurons_d1); %zhou record
     inh_pv_Vm = zeros(t_end/dt+1,inh_Nneurons_pv); %zhou record
     inh_cb_Vm = zeros(t_end/dt+1,inh_Nneurons_cb); %zhou record
%      pyr_s_x_NMDA_record = zeros(t_end/dt+1,pyr_Nneurons_s); %zhou 
%      pyr_s_s_NMDA_record  = zeros(t_end/dt+1,pyr_Nneurons_s); %zhou record
%      pyr_s_s_AMPA_record = zeros(t_end/dt+1,pyr_Nneurons_s); %zhou record
%      pyr_d1_x_NMDA_record = zeros(t_end/dt+1,pyr_Nneurons_s);  %zhou record
%      pyr_d1_s_NMDA_record  =zeros(t_end/dt+1,pyr_Nneurons_s);  %zhou record
%      pyr_d1_s_AMPA_record = zeros(t_end/dt+1,pyr_Nneurons_s);  %zhou record
%      inh_pv_s_GABA_record = zeros(t_end/dt+1,inh_Nneurons_pv);
%      inh_cb_s_GABA_record = zeros(t_end/dt+1,inh_Nneurons_cb);
%      inh_pv_I_NMDA_record=zeros(t_end/dt+1,inh_Nneurons_pv);
%     inh_pv_I_AMPA_record=zeros(t_end/dt+1,inh_Nneurons_pv);
%     inh_pv_I_GABA_pv_record=zeros(t_end/dt+1,inh_Nneurons_pv);
%     inh_pv_I_GABA_cb_record=zeros(t_end/dt+1,inh_Nneurons_pv);
    % record pyr_s_I
    load('pyr_s_I_AMPA_record_init_10ms.mat');
    pyr_s_I_AMPA_record = zeros(t_end/dt+1,pyr_Nneurons_s); %zhou record 
%     pyr_s_I_GABA_record = zeros(t_end/dt+1,pyr_Nneurons_s); %zhou record
%     pyr_s_I_NMDA_record = zeros(t_end/dt+1,pyr_Nneurons_s); %zhou record
%     pyr_d1_I_GABA_record = zeros(t_end/dt+1,pyr_Nneurons_d1); %zhou record
%     inh_cb_I_NMDA_record = zeros(t_end/dt+1,inh_Nneurons_cb); %stp 10/19
%     pyr_s_I_AMPA_sum = zeros(1,pyr_Nneurons_s); %new record 12/28
%     pyr_s_I_GABA_sum = zeros(1,pyr_Nneurons_s); %new record 12/28
%     pyr_s_I_NMDA_sum = zeros(1,pyr_Nneurons_s); %new record 12/28
%     pyr_d1_I_GABA_sum = zeros(1,pyr_Nneurons_s); %new record 12/28
    
    % record the u of each neuron %stp 2018/10/12
%     pyr_s_u = zeros(t_end/dt+1,pyr_Nneurons_s);

 % Connectivity vectors:
%% exc projection
% ** Connectivity vector for recurrent s->s connections
    pref_dirs_pop_pyr_s = (0:360/pyr_Nneurons_s:(360-360/pyr_Nneurons_s))';
    pref_dir_diff_pyr_s = min(pref_dirs_pop_pyr_s,360-pref_dirs_pop_pyr_s);
    gauss_con = exp(-0.5*pref_dir_diff_pyr_s.^2/sigma_pyr_s2pyr_s^2);
    sigma_pyr_s2pyr_s = sigma_pyr_s2pyr_s/360;
    tmp2=sqrt(2*pi)*sigma_pyr_s2pyr_s*erf(.5/sqrt(2)/sigma_pyr_s2pyr_s);
    tmp1=(1.-tmp2*J_plus_pyr_s2pyr_s)/(1.-tmp2);
    con_vec_not_normalized = tmp1+(J_plus_pyr_s2pyr_s-tmp1)*gauss_con;
    con_vec = con_vec_not_normalized/sum(con_vec_not_normalized);
    W_pyr_s2pyr_s_fft = fft(con_vec);
    
    clear gauss_con tmp2 tmp1 con_vec_not_normalized con_vec
 
% ** Connectivity vector for recurrent s->pv connections:no
    pref_dirs_pop_inh_pv = (0:360/inh_Nneurons_pv:(360-360/inh_Nneurons_pv))';
    pref_dir_diff_pyr_s2inh_pv = repmat(pref_dirs_pop_pyr_s',inh_Nneurons_pv,1) - repmat(pref_dirs_pop_inh_pv,1,pyr_Nneurons_s);
    pref_dir_diff_pyr_s2inh_pv = min(abs(pref_dir_diff_pyr_s2inh_pv),360-abs(pref_dir_diff_pyr_s2inh_pv));
    gauss_con_pyr_s2inh_pv = exp(-0.5*pref_dir_diff_pyr_s2inh_pv.^2/sigma_pyr_s2inh_pv^2);
    J_minus_pyr_s2inh_pv = (360-sqrt(2*pi)*sigma_pyr_s2inh_pv*J_plus_pyr_s2inh_pv)/(360-sqrt(2*pi)*sigma_pyr_s2inh_pv);
    con_vec_not_normalized_pyr_s2inh_pv = J_minus_pyr_s2inh_pv+(J_plus_pyr_s2inh_pv-J_minus_pyr_s2inh_pv)*gauss_con_pyr_s2inh_pv;
    con_vec_pyr_s2inh_pv = zeros(inh_Nneurons_pv,pyr_Nneurons_s);
    for i = 1:inh_Nneurons_pv
        con_vec_pyr_s2inh_pv(i,:) = con_vec_not_normalized_pyr_s2inh_pv(i,:)/sum(con_vec_not_normalized_pyr_s2inh_pv(i,:)).*pyr_Nneurons_s;
    end
    W_pyr_s2inh_pv = con_vec_pyr_s2inh_pv;
    
    clear gauss_con_pyr_s2inh_pv con_vec_not_normalized_pyr_s2inh_pv con_vec_pyr_s2inh_pv
    
% ** Connectivity vector for recurrent s->cb connections:no
    pref_dirs_pop_inh_cb = (0:360/inh_Nneurons_cb:(360-360/inh_Nneurons_cb))';
    pref_dir_diff_pyr_s2inh_cb = repmat(pref_dirs_pop_pyr_s',inh_Nneurons_cb,1) - repmat(pref_dirs_pop_inh_cb,1,pyr_Nneurons_s);
    pref_dir_diff_pyr_s2inh_cb = min(abs(pref_dir_diff_pyr_s2inh_cb),360-abs(pref_dir_diff_pyr_s2inh_cb));
    gauss_con_pyr_s2inh_cb = exp(-0.5*pref_dir_diff_pyr_s2inh_cb.^2/sigma_pyr_s2inh_cb^2);%ones(size(pref_dir_diff_pyr_s2inh_cb));gausscb%exp(-0.5*pref_dir_diff_pyr_s2inh_cb.^2/sigma_pyr_s2inh_cb^2);
    J_minus_pyr_s2inh_cb = (360-sqrt(2*pi)*sigma_pyr_s2inh_cb*J_plus_pyr_s2inh_cb)/(360-sqrt(2*pi)*sigma_pyr_s2inh_cb);
    con_vec_not_normalized_pyr_s2inh_cb = J_minus_pyr_s2inh_cb+(J_plus_pyr_s2inh_cb-J_minus_pyr_s2inh_cb)*gauss_con_pyr_s2inh_cb;
    con_vec_pyr_s2inh_cb = zeros(inh_Nneurons_cb,pyr_Nneurons_s);
    for i = 1:inh_Nneurons_cb
        con_vec_pyr_s2inh_cb(i,:) = con_vec_not_normalized_pyr_s2inh_cb(i,:)/sum(con_vec_not_normalized_pyr_s2inh_cb(i,:)).*pyr_Nneurons_s;
    end
    W_pyr_s2inh_cb = con_vec_pyr_s2inh_cb;

    clear gauss_con_pyr_s2inh_cb con_vec_not_normalized_pyr_s2inh_cb con_vec_pyr_s2inh_cb
    
 %% inh projection
% ** Connectivity vector for recurrent cb->d1 connections
    pref_dirs_pop_pyr_d1 = (0:360/pyr_Nneurons_d1:(360-360/pyr_Nneurons_d1))';
    pref_dir_diff_inh_cb2pyr_d1 = repmat(pref_dirs_pop_inh_cb',pyr_Nneurons_d1,1) - repmat(pref_dirs_pop_pyr_d1,1,inh_Nneurons_cb);
    pref_dir_diff_inh_cb2pyr_d1 = min(abs(pref_dir_diff_inh_cb2pyr_d1),360-abs(pref_dir_diff_inh_cb2pyr_d1));
    gauss_con_inh_cb2pyr_d1 = exp(-0.5*pref_dir_diff_inh_cb2pyr_d1.^2/sigma_inh_cb2pyr_d1^2);
    con_vec_not_normalized_inh_cb2pyr_d1 = gauss_con_inh_cb2pyr_d1;
    con_vec_inh_cb2pyr_d1 = zeros(pyr_Nneurons_d1,inh_Nneurons_cb);
    for i = 1:pyr_Nneurons_d1
        con_vec_inh_cb2pyr_d1(i,:) = con_vec_not_normalized_inh_cb2pyr_d1(i,:)/sum(con_vec_not_normalized_inh_cb2pyr_d1(i,:)).*inh_Nneurons_cb;
    end
    W_inh_cb2pyr_d1 = con_vec_inh_cb2pyr_d1;
    
    clear gauss_con_inh_cb2pyr_d1 con_vec_not_normalized_inh_cb2pyr_d1 con_vec_inh_cb2pyr_d1
    
% ** Connectivity vector for recurrent cb->pv connections
    pref_dirs_pop_inh_pv = (0:360/inh_Nneurons_pv:(360-360/inh_Nneurons_pv))';
    pref_dir_diff_inh_cb2inh_pv = repmat(pref_dirs_pop_inh_cb', inh_Nneurons_pv,1) - repmat(pref_dirs_pop_inh_pv,1,inh_Nneurons_cb);
    pref_dir_diff_inh_cb2inh_pv = min(abs(pref_dir_diff_inh_cb2inh_pv),360-abs(pref_dir_diff_inh_cb2inh_pv));
    gauss_con_inh_cb2inh_pv = ones(size(pref_dir_diff_inh_cb2inh_pv));%exp(-0.5*pref_dir_diff_inh_pv2inh_cb.^2/sigma_inh_pv2inh_cb^2);
    con_vec_not_normalized_inh_cb2inh_pv = gauss_con_inh_cb2inh_pv;
%     con_vec_inh_cb2inh_pv = zeros(inh_Nneurons_pv,inh_Nneurons_cb);
%     for i = 1: inh_Nneurons_pv
%         con_vec_inh_cb2inh_pv(i,:) = con_vec_not_normalized_inh_cb2inh_pv(i,:)/sum(con_vec_not_normalized_inh_cb2inh_pv(i,:)).*inh_Nneurons_cb;
%     end
    con_vec_inh_cb2inh_pv = con_vec_not_normalized_inh_cb2inh_pv/sum(con_vec_not_normalized_inh_cb2inh_pv);
    W_inh_cb2inh_pv_fft = fft(con_vec_inh_cb2inh_pv);

    clear gauss_con_inh_cb2inh_pv con_vec_not_normalized_inh_cb2inh_pv con_vec_inh_cb2inh_pv
    
% ** Connectivity vector for recurrent pv->pv connections:no
    pref_dirs_pop_inh_pv = (0:360/inh_Nneurons_pv:(360-360/inh_Nneurons_pv))';
    pref_dir_diff_inh_pv = min(pref_dirs_pop_inh_pv,360-pref_dirs_pop_inh_pv);
    gauss_con = ones(size(pref_dir_diff_inh_pv));%exp(-0.5*pref_dir_diff_inh_pv.^2/sigma_inh_pv2inh_pv^2);
    con_vec_not_normalized = gauss_con;
    con_vec = con_vec_not_normalized/sum(con_vec_not_normalized);
    W_inh_pv2inh_pv_fft = fft(con_vec);
    
    clear gauss_con con_vec_not_normalized con_vec
    
% ** Connectivity vector for recurrent pv->s connections
    pref_dirs_pop_pyr_s = (0:360/pyr_Nneurons_s:(360-360/pyr_Nneurons_s))';
    pref_dir_diff_inh_pv2pyr_s = repmat(pref_dirs_pop_inh_pv', pyr_Nneurons_s,1) - repmat(pref_dirs_pop_pyr_s,1,inh_Nneurons_pv);
    pref_dir_diff_inh_pv2pyr_s = min(abs(pref_dir_diff_inh_pv2pyr_s),360-abs(pref_dir_diff_inh_pv2pyr_s));
    gauss_con_inh_pv2pyr_s = exp(-0.5*pref_dir_diff_inh_pv2pyr_s.^2/sigma_inh_pv2pyr_s^2);
    con_vec_not_normalized_inh_pv2pyr_s = gauss_con_inh_pv2pyr_s;
    con_vec_inh_pv2pyr_s = zeros(pyr_Nneurons_s,inh_Nneurons_pv);
    for i = 1:pyr_Nneurons_s
        con_vec_inh_pv2pyr_s(i,:) = con_vec_not_normalized_inh_pv2pyr_s(i,:)/sum(con_vec_not_normalized_inh_pv2pyr_s(i,:)).*inh_Nneurons_pv;
    end
    W_inh_pv2pyr_s = con_vec_inh_pv2pyr_s;
    
    clear gauss_con_inh_pv2pyr_s con_vec_not_normalized_inh_pv2pyr_s con_vec_inh_pv2pyr_s
    
% ** Connectivity vector for recurrent pv->cb connections:no
    pref_dirs_pop_inh_cb = (0:360/inh_Nneurons_cb:(360-360/inh_Nneurons_cb))';
    pref_dir_diff_inh_pv2inh_cb = repmat(pref_dirs_pop_inh_pv', inh_Nneurons_cb,1) - repmat(pref_dirs_pop_inh_cb,1,inh_Nneurons_pv);
    pref_dir_diff_inh_pv2inh_cb = min(abs(pref_dir_diff_inh_pv2inh_cb),360-abs(pref_dir_diff_inh_pv2inh_cb));
    gauss_con_inh_pv2inh_cb = exp(-0.5*pref_dir_diff_inh_pv2inh_cb.^2/sigma_inh_pv2inh_cb^2);%ones(size(pref_dir_diff_inh_pv2inh_cb));gausspv-cb%exp(-0.5*pref_dir_diff_inh_pv2inh_cb.^2/sigma_inh_pv2inh_cb^2);
    con_vec_not_normalized_inh_pv2inh_cb = gauss_con_inh_pv2inh_cb;
    con_vec_inh_pv2inh_cb = zeros(inh_Nneurons_cb,inh_Nneurons_pv);
    for i = 1: inh_Nneurons_cb
        con_vec_inh_pv2inh_cb(i,:) = con_vec_not_normalized_inh_pv2inh_cb(i,:)/sum(con_vec_not_normalized_inh_pv2inh_cb(i,:)).*inh_Nneurons_pv;
    end
%     con_vec_inh_pv2inh_cb = con_vec_not_normalized_inh_pv2inh_cb/sum(con_vec_not_normalized_inh_pv2inh_cb);
%     W_inh_pv2inh_cb_fft = fft(con_vec_inh_pv2inh_cb);
    W_inh_pv2inh_cb = con_vec_inh_pv2inh_cb;

    clear gauss_con_inh_pv2inh_cb con_vec_not_normalized_inh_pv2inh_cb con_vec_inh_pv2inh_cb
    
% Recording files:
    fname_pyr_s = [root_dir,'pyr_cell_s',num2str(datanum),'.txt'];
    fname_pyr_d1 = [root_dir,'pyr_cell_d1',num2str(datanum),'.txt'];
    fname_inh_pv = [root_dir,'inh_cell_pv',num2str(datanum),'.txt'];
    fname_inh_cb = [root_dir,'inh_cell_cb',num2str(datanum),'.txt'];
    fname_Vm_s = [root_dir,'Vm_s',num2str(datanum),'.txt'];
    fname_Vm_d1 = [root_dir,'Vm_d1',num2str(datanum),'.txt'];
    fname_Vm_pv = [root_dir,'Vm_pv',num2str(datanum),'.txt'];
    fname_Vm_cb = [root_dir,'Vm_cb',num2str(datanum),'.txt'];
    fname_s_NMDA_s = [root_dir,'s_NMDA_s',num2str(datanum),'.txt'];
    fname_s_NMDA_x = [root_dir,'s_NMDA_x',num2str(datanum),'.txt'];
    fname_s_AMPA_s = [root_dir,'s_AMPA_s',num2str(datanum),'.txt'];
    fname_d1_NMDA_s = [root_dir,'d1_NMDA_s',num2str(datanum),'.txt'];
    fname_d1_NMDA_x = [root_dir,'d1_NMDA_x',num2str(datanum),'.txt'];
    fname_d1_AMPA_s = [root_dir,'d1_AMPA_s',num2str(datanum),'.txt'];
    fname_pv_GABA_s = [root_dir,'pv_GABA_s',num2str(datanum),'.txt'];
    fname_cb_GABA_s = [root_dir,'cb_GABA_s',num2str(datanum),'.txt'];
    fname_lfp = [root_dir,'lfp',num2str(datanum),'.txt'];
    fname_lfp_rws = [root_dir,'lfp_rws',num2str(datanum),'.txt'];
    

    if exist(fname_pyr_s,'file')
        delete(fname_pyr_s)
    end
    
    if exist(fname_pyr_d1,'file')
        delete(fname_pyr_d1)
    end
    
    if exist(fname_pyr_d1,'file')
        delete(fname_pyr_d1)
    end

    if exist(fname_inh_pv,'file')
        delete(fname_inh_pv)
    end
    
    if exist(fname_inh_cb,'file')
        delete(fname_inh_cb)
    end
    
    f_pyr_s = fopen(fname_pyr_s,'a','native');
    f_pyr_d1 = fopen(fname_pyr_d1,'a','native');
    f_inh_pv = fopen(fname_inh_pv,'a','native');
    f_inh_cb = fopen(fname_inh_cb,'a','native');
    f_Vm_s = fopen(fname_Vm_s,'a','native');
    f_Vm_d1 = fopen(fname_Vm_d1,'a','native');
    f_Vm_pv = fopen(fname_Vm_pv,'a','native');
    f_Vm_cb = fopen(fname_Vm_cb,'a','native');
    f_s_NMDA_s = fopen(fname_s_NMDA_s,'a','native');
    f_s_NMDA_x = fopen(fname_s_NMDA_x,'a','native');
    f_s_AMPA_s = fopen(fname_s_AMPA_s,'a','native');
    f_d1_NMDA_s = fopen(fname_d1_NMDA_s,'a','native');
    f_d1_NMDA_x = fopen(fname_d1_NMDA_x,'a','native');
    f_d1_AMPA_s = fopen(fname_d1_AMPA_s,'a','native');
    f_pv_GABA_s = fopen(fname_pv_GABA_s,'a','native');
    f_cb_GABA_s = fopen(fname_cb_GABA_s,'a','native');
    f_lfp = fopen(fname_lfp,'a','native');
    f_lfp_rws = fopen(fname_lfp_rws,'a','native');
    

    
%%%%%%
     disp ('Setup for recoding data is done ....');
    

%%%%% simulation loop

    for current_time = 0:dt:t_end %

        if mod(current_time,100) == 0 
            disp(['Time: ',num2str(current_time),' ms']);
        end
  
        % Update indices of firing bank:
        pyr_s_popA_ind_now_in_bank = mod(pyr_s_popA_ind_now_in_bank,...
            pyr_s_popA_delay_bank_size)+1;
        pyr_d1_popA_ind_now_in_bank = mod(pyr_d1_popA_ind_now_in_bank,...
            pyr_d1_popA_delay_bank_size)+1;
        
        inh_pv_popA_ind_now_in_bank = mod(inh_pv_popA_ind_now_in_bank,...
            inh_pv_popA_delay_bank_size)+1;
        inh_cb_popA_ind_now_in_bank = mod(inh_cb_popA_ind_now_in_bank,...
            inh_cb_popA_delay_bank_size)+1;

        % Update NMDA AMPA and GABA
        % NMDA
        pyr_s_x_NMDA_new = RK2_simple_linear_eq(pyr_s_x_NMDA,dt, -1/syn_tau_x_NMDA, 0);
        pyr_s_x_NMDA_new(pyr_s_WhoFired) = pyr_s_x_NMDA_new(pyr_s_WhoFired)+syn_alpha_x_NMDA;
        pyr_s_s_NMDA_new = RK2_4sNMDA(pyr_s_s_NMDA,dt,-1/syn_tau_s_NMDA, ...
            syn_alpha_s_NMDA*pyr_s_x_NMDA);
        pyr_s_x_NMDA=pyr_s_x_NMDA_new;
        pyr_s_s_NMDA=pyr_s_s_NMDA_new;
        
        
        
        
        pyr_d1_x_NMDA_new = RK2_simple_linear_eq(pyr_d1_x_NMDA,dt, -1/syn_tau_x_NMDA, 0);
        pyr_d1_x_NMDA_new(pyr_d1_WhoFired) = pyr_d1_x_NMDA_new(pyr_d1_WhoFired)+syn_alpha_x_NMDA;
        pyr_d1_s_NMDA_new = RK2_4sNMDA(pyr_d1_s_NMDA,dt,-1/syn_tau_s_NMDA, ...
            syn_alpha_s_NMDA*pyr_d1_x_NMDA);
        pyr_d1_x_NMDA=pyr_d1_x_NMDA_new;
        pyr_d1_s_NMDA=pyr_d1_s_NMDA_new;
        
%         pyr_s_fire_NMDA = zeros(pyr_Nneurons_s,1); %stp 2018/10/12
%         pyr_s_fire_NMDA(pyr_s_WhoFired) = pyr_s_fire_NMDA(pyr_s_WhoFired)+syn_alpha_u_NMDA; %stp
        pyr_s_u_NMDA = RK2_4uNMDA(pyr_s_u_NMDA,dt,1/syn_tau_f_NMDA, ...
            0,pyr_popA_init_U_NMDA,0); %stp
        pyr_s_u_NMDA(pyr_s_WhoFired) = pyr_s_u_NMDA(pyr_s_WhoFired) + ...
            syn_U_NMDA .* (1 - pyr_s_u_NMDA(pyr_s_WhoFired));
        
        
        % AMPA
        pyr_s_s_AMPA = RK2_simple_linear_eq(pyr_s_s_AMPA,dt,-1/syn_tau_s_AMPA, 0);
        pyr_s_s_AMPA(pyr_s_WhoFired) = pyr_s_s_AMPA(pyr_s_WhoFired)+syn_alpha_s_AMPA;
        
        pyr_d1_s_AMPA = RK2_simple_linear_eq(pyr_d1_s_AMPA,dt,-1/syn_tau_s_AMPA, 0);
        pyr_d1_s_AMPA(pyr_d1_WhoFired) = pyr_d1_s_AMPA(pyr_d1_WhoFired)+syn_alpha_s_AMPA;
        
        % GABA
        inh_pv_s_GABA = RK2_simple_linear_eq(inh_pv_s_GABA,dt,-1/syn_tau_s_GABA, 0);
        inh_pv_s_GABA(inh_pv_WhoFired) = inh_pv_s_GABA(inh_pv_WhoFired)+syn_alpha_s_GABA;
        
        inh_cb_s_GABA = RK2_simple_linear_eq(inh_cb_s_GABA,dt,-1/syn_tau_s_GABA, 0);
        inh_cb_s_GABA(inh_cb_WhoFired) = inh_cb_s_GABA(inh_cb_WhoFired)+syn_alpha_s_GABA;
       
        %%%%%%%%%%%%%%%%%%%%
        % *************************
        % Pyramidal cells update:
        % *************************
        
       % Generate input signal
        pyr_profile=zeros(pyr_Nneurons_s,1);
        pyr_I_applied = 0;
     for n_stim=1:n_stimuli
            input_para = sequential_tensify(1);
        if n_stim > 1;
            input_para = sequential_tensify(2);
        end
            pyr_profile = max_stim_current*input_para*...
            circular_gaussian(pyr_Nneurons_s,sigma_stim,theta_stim(n_stim));
            pyr_input_n = pyr_input + sequential_delay.*(n_stim - 1); %calculate the input time of nth stim
            k_time_period = logical(sum(current_time<pyr_input_n))... 
            *logical(sum(current_time>=pyr_input_n))*...
            sum(current_time>=pyr_input_n); %determine whether current_time is in [pyr_input_n(1), pyr_input_n(2)]
        if k_time_period > 0
            pyr_I_applied = pyr_I_applied + pyr_profile;
        end
    end
       
% pyr_s update        

        % GABA current
        w_dot_s_GABA_inh_pv2pyr_s = W_inh_pv2pyr_s * inh_pv_s_GABA;
        pyr_s_I_GABA = popA_G_inh_pv2pyr_s_GABA.* w_dot_s_GABA_inh_pv2pyr_s.*(pyr_s_Vm_prev...
            -syn_Ve_GABA);
        

        % Update recurrent currents; thresholding etc.
        pyr_s_Vm_prev = pyr_s_Vm_new; % store voltage from the previous step
        
        % External AMPA current
        rand_vec = rand(pyr_Nneurons_s,1) < dt*pyr_ext_noise_rate;
        pyr_s_ext_noise_s_AMPA=RK2_simple_linear_eq(pyr_s_ext_noise_s_AMPA,dt,...
            -1/syn_tau_s_AMPA,syn_alpha_s_AMPA*rand_vec);
        pyr_s_I_noise=pyr_ext_noise_g_AMPA*pyr_s_ext_noise_s_AMPA.*(pyr_s_Vm_prev...
            -syn_Ve_AMPA);
        pyr_s_I_leak=pyr_g_leak*(pyr_s_Vm_prev-pyr_Vm_leak);

        % Multiply weight matrix by s vector using FFT:
        w_dot_s_NMDA_pyr_s2pyr_s=ifft(W_pyr_s2pyr_s_fft.*fft(pyr_s_s_NMDA));
        w_dot_s_AMPA_pyr_s2pyr_s=ifft(W_pyr_s2pyr_s_fft.*fft(pyr_s_s_AMPA));
        
        % NMDA current
        pyr_s_I_NMDA = popA_G_pyr_s2pyr_s_NMDA*pyr_Nneurons_s*w_dot_s_NMDA_pyr_s2pyr_s.*...
            (pyr_s_Vm_prev-syn_Ve_NMDA)./ ...
            (1+syn_Mg_concentration*exp(-0.062*pyr_s_Vm_prev)/3.57);
        % AMPA current
        pyr_s_I_AMPA = popA_G_pyr_s2pyr_s_AMPA*pyr_Nneurons_s*w_dot_s_AMPA_pyr_s2pyr_s.*...
            (pyr_s_Vm_prev-syn_Ve_AMPA);

        % high-threshold calcium current
        m_Ca_s = 1./(1 + exp(-(pyr_s_Vm_prev + 20)./9));
        pyr_s_I_Ca = g_Ca_s.* m_Ca_s.^2.* (pyr_s_Vm_prev - 120);

        % TOTAL CURRENT:
        pyr_s_I_total = - pyr_s_I_leak - pyr_s_I_NMDA - pyr_s_I_GABA - pyr_s_I_AMPA - pyr_s_I_noise...
            + pyr_I_applied - gc1.*(pyr_s_Vm_prev - pyr_d1_Vm_prev)./p1 - pyr_s_I_Ca;

        % Membrate voltage:
        pyr_s_Vm_new=pyr_s_Vm_new+1/pyr_Cm*dt*pyr_s_I_total;
        pyr_s_Vm_new((current_time-pyr_s_LastTimeEachFired)<pyr_tau_ref) = ...
            pyr_Vm_reset;
        pyr_s_WhoFiredNow=find(pyr_s_Vm_new>pyr_Vm_thresh);
        pyr_s_SpikeTimes=current_time+dt*((pyr_Vm_thresh...
            -pyr_s_Vm_prev(pyr_s_WhoFiredNow))./(pyr_s_Vm_new(pyr_s_WhoFiredNow)...
            -pyr_s_Vm_prev(pyr_s_WhoFiredNow)));
        pyr_s_Vm_new(pyr_s_WhoFiredNow)=pyr_Vm_reset;
        pyr_s_LastTimeEachFired(pyr_s_WhoFiredNow)=pyr_s_SpikeTimes;
        
        

        %%%% Firing and transmission delays:
        pyr_s_WhoFired = find(pyr_s_popA_WhoFired_bank(:,...
            pyr_s_popA_ind_now_in_bank));
        pyr_s_popA_WhoFired_bank(:,pyr_s_popA_ind_now_in_bank) = 0;

        non_circular_indices4storage = pyr_s_popA_ind_now_in_bank + ...
            pyr_s_popA_transmission_delay_t_steps(pyr_s_WhoFiredNow);

        circular_indices4storage = mod(non_circular_indices4storage-1,...
            pyr_s_popA_delay_bank_size)+1;
        linear_indices4storage = pyr_s_WhoFiredNow + ...
            (circular_indices4storage - 1)*pyr_Nneurons_s;
        pyr_s_popA_WhoFired_bank(linear_indices4storage) = 1;
        

        % Store spikes
        n_fired_s = size(pyr_s_WhoFiredNow,1);
        for k_fired_s=1:n_fired_s
            fprintf(f_pyr_s,'%f,%f\n',pyr_s_WhoFiredNow(k_fired_s),...
                pyr_s_SpikeTimes(k_fired_s));
        end
%firing rate
        fr(round(int32(current_time/dt+1)/50)+1,pyr_s_WhoFiredNow)=fr(round(int32(current_time/dt+1)/50)+1,pyr_s_WhoFiredNow)+1;
        fr_1(round(int32(current_time/dt+1)),pyr_s_WhoFiredNow)=1;
% pyr_d1 update
        % GABA current
        w_dot_s_GABA_inh_cb2pyr_d1 = W_inh_cb2pyr_d1 * inh_cb_s_GABA;
        pyr_d1_I_GABA = popA_G_inh_cb2pyr_d1_GABA.* w_dot_s_GABA_inh_cb2pyr_d1.* (pyr_d1_Vm_prev...
            -syn_Ve_GABA);
        

        
        % Update recurrent currents; thresholding etc.
        pyr_d1_Vm_prev = pyr_d1_Vm_new; % store voltage from the previous step
        
        % External AMPA current
        rand_vec = rand(pyr_Nneurons_d1,1) < dt*pyr_ext_noise_rate;
        pyr_d1_ext_noise_s_AMPA=RK2_simple_linear_eq(pyr_d1_ext_noise_s_AMPA,dt,...
            -1/syn_tau_s_AMPA,syn_alpha_s_AMPA*rand_vec);
        pyr_d1_I_noise=pyr_ext_noise_g_AMPA*pyr_d1_ext_noise_s_AMPA.*(pyr_d1_Vm_prev...
            -syn_Ve_AMPA);
        pyr_d1_I_leak=pyr_g_leak_d1*(pyr_d1_Vm_prev-pyr_Vm_leak);

        % TOTAL CURRENT:
        pyr_d1_I_total = - pyr_d1_I_leak - pyr_d1_I_GABA - pyr_d1_I_noise...
            + pyr_I_applied - gc1.*(pyr_d1_Vm_prev - pyr_s_Vm_prev)./p2; %no d2

        % Membrate voltage:
        pyr_d1_Vm_new=pyr_d1_Vm_new+1/pyr_Cm*dt*pyr_d1_I_total;
        pyr_d1_Vm_new((current_time-pyr_d1_LastTimeEachFired)<pyr_tau_ref) = ...
            pyr_Vm_reset;
        pyr_d1_WhoFiredNow=find(pyr_d1_Vm_new>pyr_Vm_thresh);
        pyr_d1_SpikeTimes=current_time+dt*((pyr_Vm_thresh...
            -pyr_d1_Vm_prev(pyr_d1_WhoFiredNow))./(pyr_d1_Vm_new(pyr_d1_WhoFiredNow)...
            -pyr_d1_Vm_prev(pyr_d1_WhoFiredNow)));
        pyr_d1_Vm_new(pyr_d1_WhoFiredNow)=pyr_Vm_reset;
        pyr_d1_LastTimeEachFired(pyr_d1_WhoFiredNow)=pyr_d1_SpikeTimes;

        %%%% Firing and transmission delays:
        pyr_d1_WhoFired = find(pyr_d1_popA_WhoFired_bank(:,...
            pyr_d1_popA_ind_now_in_bank));
        pyr_d1_popA_WhoFired_bank(:,pyr_d1_popA_ind_now_in_bank) = 0;

        non_circular_indices4storage = pyr_d1_popA_ind_now_in_bank + ...
            pyr_d1_popA_transmission_delay_t_steps(pyr_d1_WhoFiredNow);

        circular_indices4storage = mod(non_circular_indices4storage-1,...
            pyr_d1_popA_delay_bank_size)+1;
        linear_indices4storage = pyr_d1_WhoFiredNow + ...
            (circular_indices4storage - 1)*pyr_Nneurons_d1;
        pyr_d1_popA_WhoFired_bank(linear_indices4storage) = 1;
        

        % Store spikes

        n_fired_d1 = size(pyr_d1_WhoFiredNow,1);
        for k_fired_d1=1:n_fired_d1
            fprintf(f_pyr_d1,'%f,%f\n',pyr_d1_WhoFiredNow(k_fired_d1),...
                pyr_d1_SpikeTimes(k_fired_d1));
        end

        
        %%%%%%%%%%%%%%%%%%%%
        % *************************
        % Interneurons update:
        % *************************

% inh_pv update
        
% Multiply weight matrix by s vector using gauss:
        w_dot_s_NMDA_pyr_s2inh_pv=W_pyr_s2inh_pv*pyr_s_s_NMDA;
        w_dot_s_AMPA_pyr_s2inh_pv=W_pyr_s2inh_pv*pyr_s_s_AMPA;
        % Multiply weight matrix by s vector using FFT:
        w_dot_s_GABA_inh_pv2inh_pv=ifft(W_inh_pv2inh_pv_fft.*fft(inh_pv_s_GABA));
        w_dot_s_GABA_inh_cb2inh_pv=ifft(W_inh_cb2inh_pv_fft.*fft(inh_cb_s_GABA));
       
        % Recurrent currents
        inh_pv_I_NMDA = popA_G_pyr_s2inh_pv_NMDA.* w_dot_s_NMDA_pyr_s2inh_pv.*...
            (inh_pv_Vm_prev - syn_Ve_NMDA)./(1 + syn_Mg_concentration...
            *exp(-0.062*inh_pv_Vm_prev)/3.57);
        inh_pv_I_AMPA = popA_G_pyr_s2inh_pv_AMPA.* w_dot_s_AMPA_pyr_s2inh_pv.*...
            (inh_pv_Vm_prev-syn_Ve_AMPA);
        inh_pv_I_GABA_pv = popA_G_inh_pv2inh_pv_GABA.* w_dot_s_GABA_inh_pv2inh_pv.* (inh_pv_Vm_prev...
            -syn_Ve_GABA);
        inh_pv_I_GABA_cb = popA_G_inh_cb2inh_pv_GABA.* w_dot_s_GABA_inh_cb2inh_pv.* (inh_pv_Vm_prev...
            -syn_Ve_GABA);
        
%         inh_pv_I_NMDA_record(int32(current_time/dt+1),:)=inh_pv_I_NMDA;
%         inh_pv_I_AMPA_record(int32(current_time/dt+1),:)=inh_pv_I_NMDA;
%         inh_pv_I_GABA_pv_record(int32(current_time/dt+1),:)=inh_pv_I_NMDA;
%         inh_pv_I_GABA_cb_record(int32(current_time/dt+1),:)=inh_pv_I_NMDA;
     
       
        

        %%%%%%%%%%%%%%%%%%%%%%%%%

        % Update recurrent currents; thresholding etc.
        inh_pv_Vm_prev=inh_pv_Vm_new; % store voltage from the previous step
        % External AMPA current
        rand_vec=rand(inh_Nneurons_pv,1)<dt*inh_ext_noise_rate;
        inh_pv_ext_noise_s_AMPA=RK2_simple_linear_eq(inh_pv_ext_noise_s_AMPA,...
            dt,-1/syn_tau_s_AMPA,syn_alpha_s_AMPA*rand_vec);
        inh_pv_I_noise=inh_pv_ext_noise_g_AMPA*inh_pv_ext_noise_s_AMPA.*...
            (inh_pv_Vm_prev-syn_Ve_AMPA);
        inh_pv_I_leak=inh_g_leak_pv*(inh_pv_Vm_prev-inh_Vm_leak_pv);

        % TOTAL CURRENT:
        inh_pv_I_total = - inh_pv_I_leak - inh_pv_I_NMDA - inh_pv_I_GABA_pv -...
            inh_pv_I_GABA_cb - inh_pv_I_AMPA - inh_pv_I_noise;%zhou
        pv_I_syn= - inh_pv_I_NMDA - inh_pv_I_GABA_pv -...
            inh_pv_I_GABA_cb - inh_pv_I_AMPA;
        
        FS_prefer_total_I(int32(current_time/dt+1))=mean(pv_I_syn(246:266));
        FS_nonprefer_total_I(int32(current_time/dt+1))=mean(mean([pv_I_syn(1:11),pv_I_syn(502:512)]));

        % Membrate voltage:
        inh_pv_Vm_new=inh_pv_Vm_new+1/inh_Cm_pv*dt*inh_pv_I_total;
        inh_pv_Vm_new((current_time-inh_pv_LastTimeEachFired)<inh_tau_ref)=...
            inh_Vm_reset;
        inh_pv_WhoFiredNow=find(inh_pv_Vm_new>inh_Vm_thresh);
        inh_pv_SpikeTimes=current_time+dt*((inh_Vm_thresh-...
            inh_pv_Vm_prev(inh_pv_WhoFiredNow))./(inh_pv_Vm_new(inh_pv_WhoFiredNow)-...
            inh_pv_Vm_prev(inh_pv_WhoFiredNow)));
        inh_pv_Vm_new(inh_pv_WhoFiredNow)=inh_Vm_reset;
        inh_pv_LastTimeEachFired(inh_pv_WhoFiredNow)=inh_pv_SpikeTimes;    
        
        % Firing and transmission delays:
        inh_pv_WhoFired = find(inh_pv_popA_WhoFired_bank(:,inh_pv_popA_ind_now_in_bank));
        inh_pv_popA_WhoFired_bank(:,inh_pv_popA_ind_now_in_bank) = 0;

        non_circular_indices4storage = ...
            inh_pv_popA_ind_now_in_bank + ...
            inh_pv_popA_transmission_delay_t_steps(inh_pv_WhoFiredNow);

        circular_indices4storage = mod(non_circular_indices4storage-1,...
            inh_pv_popA_delay_bank_size)+1;
        linear_indices4storage = inh_pv_WhoFiredNow+(circular_indices4storage...
            - 1)*inh_Nneurons_pv;
        inh_pv_popA_WhoFired_bank(linear_indices4storage) = 1;

        % Store spikes:
        n_fired_pv = size(inh_pv_WhoFiredNow,1);
        for k_fired_pv=1:n_fired_pv
            fprintf(f_inh_pv,'%f,%f\n',inh_pv_WhoFiredNow(k_fired_pv),...
                inh_pv_SpikeTimes(k_fired_pv));
        end
        
        fr1(round(int32(current_time/dt+1)/50)+1,inh_pv_WhoFiredNow)=fr1(round(int32(current_time/dt+1)/50)+1,inh_pv_WhoFiredNow)+1;

% inh_cb update
        % Multiply weight matrix by s vector using gauss:
        w_dot_s_NMDA_pyr_s2inh_cb=W_pyr_s2inh_cb*pyr_s_s_NMDA;
        w_dot_s_AMPA_pyr_s2inh_cb=W_pyr_s2inh_cb*pyr_s_s_AMPA;
        % Multiply weight matrix by s vector using FFT:
        w_dot_s_GABA_inh_pv2inh_cb=W_inh_pv2inh_cb*inh_pv_s_GABA;
        
        % Recurrent currents %stp 2018/10/12
        w_dot_u_NMDA_pyr_s = W_pyr_s2inh_cb*(pyr_s_u_NMDA.*pyr_s_s_NMDA); %stp 2018/10/12
        inh_cb_I_NMDA = popA_G_pyr_s2inh_cb_NMDA.*w_dot_u_NMDA_pyr_s.*...
            (inh_cb_Vm_prev - syn_Ve_NMDA)./(1 + syn_Mg_concentration...
            *exp(-0.062*inh_cb_Vm_prev)/3.57); %stp 2018/10/12
        inh_cb_I_AMPA = popA_G_pyr_s2inh_cb_AMPA.* w_dot_s_AMPA_pyr_s2inh_cb.*...
            (inh_cb_Vm_prev-syn_Ve_AMPA);
        inh_cb_I_GABA = popA_G_inh_pv2inh_cb_GABA.* w_dot_s_GABA_inh_pv2inh_cb.* (inh_cb_Vm_prev...
            -syn_Ve_GABA);
        


        %%%%%%%%%%%%%%%%%%%%%%%%%

        % Update recurrent currents; thresholding etc.
        inh_cb_Vm_prev=inh_cb_Vm_new; % store voltage from the previous step
        % External AMPA current
        rand_vec=rand(inh_Nneurons_cb,1)<dt*inh_ext_noise_rate;
        inh_cb_ext_noise_s_AMPA=RK2_simple_linear_eq(inh_cb_ext_noise_s_AMPA,...
            dt,-1/syn_tau_s_AMPA,syn_alpha_s_AMPA*rand_vec);
        inh_cb_I_noise=inh_cb_ext_noise_g_AMPA*inh_cb_ext_noise_s_AMPA.*...
            (inh_cb_Vm_prev-syn_Ve_AMPA);
        inh_cb_I_leak=inh_g_leak_cb*(inh_cb_Vm_prev-inh_Vm_leak_cb);
        
        % high-threshold calcium current
        m_Ca_cb = 1./(1 + exp(-(inh_cb_Vm_prev + 20)./9));
        inh_cb_I_Ca = g_Ca_cb.* m_Ca_cb.^2.* (inh_cb_Vm_prev - 120);

        % TOTAL CURRENT:
        inh_cb_I_total = - inh_cb_I_leak - inh_cb_I_NMDA - inh_cb_I_GABA - inh_cb_I_AMPA...
            - inh_cb_I_noise - inh_cb_I_Ca;%zhou
        
        cb_I_syn= - inh_cb_I_NMDA - inh_cb_I_GABA - inh_cb_I_AMPA;
        
        NFS_prefer_total_I(int32(current_time/dt+1))=mean(cb_I_syn(246:266));
        NFS_nonprefer_total_I(int32(current_time/dt+1))=mean(mean([cb_I_syn(1:11),cb_I_syn(502:512)]));

        % Membrate voltage:
        inh_cb_Vm_new=inh_cb_Vm_new+1/inh_Cm_cb*dt*inh_cb_I_total;
        inh_cb_Vm_new((current_time-inh_cb_LastTimeEachFired)<inh_tau_ref)=...
            inh_Vm_reset;
        inh_cb_WhoFiredNow=find(inh_cb_Vm_new>inh_Vm_thresh);
        inh_cb_SpikeTimes=current_time+dt*((inh_Vm_thresh-...
            inh_cb_Vm_prev(inh_cb_WhoFiredNow))./(inh_cb_Vm_new(inh_cb_WhoFiredNow)-...
            inh_cb_Vm_prev(inh_cb_WhoFiredNow)));
        inh_cb_Vm_new(inh_cb_WhoFiredNow)=inh_Vm_reset;
        inh_cb_LastTimeEachFired(inh_cb_WhoFiredNow)=inh_cb_SpikeTimes;    
 
        % Firing and transmission delays:
        inh_cb_WhoFired = find(inh_cb_popA_WhoFired_bank(:,inh_cb_popA_ind_now_in_bank));
        inh_cb_popA_WhoFired_bank(:,inh_cb_popA_ind_now_in_bank) = 0;
 
        non_circular_indices4storage = ...
            inh_cb_popA_ind_now_in_bank + ...
            inh_cb_popA_transmission_delay_t_steps(inh_cb_WhoFiredNow);
 
        circular_indices4storage = mod(non_circular_indices4storage-1,...
            inh_cb_popA_delay_bank_size)+1;
        linear_indices4storage = inh_cb_WhoFiredNow+(circular_indices4storage...
            - 1)*inh_Nneurons_cb;
        inh_cb_popA_WhoFired_bank(linear_indices4storage) = 1;
 
        % Store spikes:
        n_fired_cb = size(inh_cb_WhoFiredNow,1);
        for k_fired_cb=1:n_fired_cb
            fprintf(f_inh_cb,'%f,%f\n',inh_cb_WhoFiredNow(k_fired_cb),...
                inh_cb_SpikeTimes(k_fired_cb));
        end 
        
        pf=2048;
        pf1=256;
        npf=1;
        npf1=1;
        fprintf(f_Vm_s,'%f,%f\n',pyr_s_Vm_new(pf),...
                pyr_s_Vm_new(npf));
        fprintf(f_Vm_d1,'%f,%f\n',pyr_d1_Vm_new(pf),...
                pyr_d1_Vm_new(npf));
        fprintf(f_Vm_pv,'%f,%f\n',inh_pv_Vm_new(pf1),...
                inh_pv_Vm_new(npf1));
        fprintf(f_Vm_cb,'%f,%f\n',inh_cb_Vm_new(pf1),...
            inh_cb_Vm_new(npf1));
        
        fprintf(f_s_NMDA_s,'%f,%f\n',pyr_s_s_NMDA_new(pf),...
            pyr_s_s_NMDA_new(npf));
        fprintf(f_s_NMDA_x,'%f,%f\n',pyr_s_x_NMDA_new(pf),...
            pyr_s_x_NMDA_new(npf));
        fprintf(f_s_AMPA_s,'%f,%f\n',pyr_s_s_AMPA(pf),...
            pyr_s_s_AMPA(npf));
        fprintf(f_d1_NMDA_s,'%f,%f\n',pyr_d1_s_NMDA_new(pf),...
            pyr_d1_s_NMDA_new(npf));
        fprintf(f_d1_NMDA_x,'%f,%f\n',pyr_d1_x_NMDA_new(pf),...
            pyr_d1_x_NMDA_new(npf));
        fprintf(f_d1_AMPA_s,'%f,%f\n',pyr_d1_s_AMPA(pf),...
            pyr_d1_s_AMPA(npf));
        fprintf(f_pv_GABA_s,'%f,%f\n',inh_pv_s_GABA(pf1),...
            inh_pv_s_GABA(npf1));
        fprintf(f_cb_GABA_s,'%f,%f\n',inh_cb_s_GABA(pf1),...
            inh_cb_s_GABA(npf1));
        
        
        
        % local field potentials
        lfp(int32(current_time/dt+1)) = mean(pyr_s_Vm_new(:)); %zhou lfp
        
        %PLoS
        if current_time < syn_tau_s_AMPA
            temp = size(pyr_s_I_AMPA_record_init,1) + (current_time-syn_tau_s_AMPA)/dt+1;
            lfp_rws(int32(current_time/dt+1)) = sum(abs(pyr_s_I_AMPA_record_init(int32(temp),:)') +...
                1.65 * abs(pyr_s_I_GABA + pyr_d1_I_GABA)); %zhou lfp %add cb2d1 pyr_d1_I_GABA
        else
            lfp_rws(int32(current_time/dt+1)) = sum(abs(pyr_s_I_AMPA_record(int32((current_time-syn_tau_s_AMPA)/dt+1))') +...
                1.65 * abs(pyr_s_I_GABA + pyr_d1_I_GABA)); %zhou lfp %add cb2d1 pyr_d1_I_GABA
        end
        
        %PLoS for preferred direction  %preferred direction
                if current_time < syn_tau_s_AMPA
            temp = size(pyr_s_I_AMPA_record_init,1) + (current_time-syn_tau_s_AMPA)/dt+1;
            lfp_rws_pre(int32(current_time/dt+1)) = sum(abs(pyr_s_I_AMPA_record_init(int32(temp),1025:3072)') +...
                1.65 * abs(pyr_s_I_GABA(1025:3072) + pyr_d1_I_GABA(1025:3072))); %zhou lfp %add cb2d1 pyr_d1_I_GABA
        else
            lfp_rws_pre(int32(current_time/dt+1)) = sum(abs(pyr_s_I_AMPA_record(int32((current_time-syn_tau_s_AMPA)/dt+1),1025:3072)') +...
                1.65 * abs(pyr_s_I_GABA(1025:3072) + pyr_d1_I_GABA(1025:3072))); %zhou lfp %add cb2d1 pyr_d1_I_GABA
        end
%         
%         if current_time < syn_tau_s_AMPA
%             temp = size(pyr_s_I_AMPA_record_init,1) + (current_time-syn_tau_s_AMPA)/dt+1;
%             lfp_rws_nonpre(int32(current_time/dt+1)) = sum(abs(pyr_s_I_AMPA_record_init(int32(temp),1:101)') +...
%                 1.65 * abs(pyr_s_I_GABA(1:101) + pyr_d1_I_GABA(1:101)))+sum(abs(pyr_s_I_AMPA_record_init(int32(temp),3996:4096)') +...
%                 1.65 * abs(pyr_s_I_GABA(3996:4096) + pyr_d1_I_GABA(3996:4096))); %zhou lfp %add cb2d1 pyr_d1_I_GABA
%         else
%             lfp_rws_nonpre(int32(current_time/dt+1)) = sum(abs(pyr_s_I_AMPA_record(int32((current_time-syn_tau_s_AMPA)/dt+1),1:101)') +...
%                 1.65 * abs(pyr_s_I_GABA(1:101) + pyr_d1_I_GABA(1:101)))+sum(abs(pyr_s_I_AMPA_record(int32((current_time-syn_tau_s_AMPA)/dt+1),3996:4096)') +...
%                 1.65 * abs(pyr_s_I_GABA(3996:4096) + pyr_d1_I_GABA(3996:4096))); %zhou lfp %add cb2d1 pyr_d1_I_GABA
%         end
        
        
        %IEEE
        lfp_ieee(int32(current_time/dt+1)) = sum(pyr_s_I_NMDA + pyr_s_I_GABA + pyr_s_I_AMPA + pyr_d1_I_GABA, 1); %add cb2d1 pyr_d1_I_GABA

        % record the Vm of each neuron
         pyr_s_Vm(int32(current_time/dt+1),:) = pyr_s_Vm_new; %zhou record
         pyr_d1_Vm(int32(current_time/dt+1),:) = pyr_d1_Vm_new; %zhou record
         inh_pv_Vm(int32(current_time/dt+1),:) = inh_pv_Vm_new; %zhou record
         inh_cb_Vm(int32(current_time/dt+1),:) = inh_cb_Vm_new; %zhou record
%          pyr_s_x_NMDA_record(int32(current_time/dt+1),:) = pyr_s_x_NMDA_new; %zhou record
%          pyr_s_s_NMDA_record(int32(current_time/dt+1),:)  = pyr_s_s_NMDA_new; %zhou record
%          pyr_s_s_AMPA_record(int32(current_time/dt+1),:) = pyr_s_s_AMPA; %zhou record
%          pyr_d1_x_NMDA_record(int32(current_time/dt+1),:) = pyr_d1_x_NMDA_new; %zhou record
%          pyr_d1_s_NMDA_record(int32(current_time/dt+1),:)  = pyr_d1_s_NMDA_new; %zhou record
%          pyr_d1_s_AMPA_record(int32(current_time/dt+1),:) = pyr_d1_s_AMPA; %zhou record
%          inh_pv_s_GABA_record(int32(current_time/dt+1),:)= inh_pv_s_GABA;
%          inh_cb_s_GABA_record(int32(current_time/dt+1),:)= inh_cb_s_GABA;
         
         fprintf(f_lfp,'%f\n',lfp(int32(current_time/dt+1)));
         fprintf(f_lfp_rws,'%f\n',lfp_rws(int32(current_time/dt+1)));
       

        % record the u of each neuron %stp 2018/10/12
%         pyr_s_u(int32(current_time/dt+1),:) = pyr_s_u_NMDA;
	
        
        % record pyr_s_I & pyr_d1_I
        pyr_s_I_AMPA_record(int32(current_time/dt+1),:) = pyr_s_I_AMPA; %zhou record
%         pyr_s_I_GABA_record(int32(current_time/dt+1),:) = pyr_s_I_GABA; %zhou record
%         pyr_s_I_NMDA_record(int32(current_time/dt+1),:) = pyr_s_I_NMDA; %zhou record
%         pyr_d1_I_GABA_record(int32(current_time/dt+1),:) = pyr_d1_I_GABA; %zhou record
        
        % record inh_pv_I & inh_cb_I
%         inh_cb_I_NMDA_record(int32(current_time/dt+1),:) = inh_cb_I_NMDA; %stp 10/19
        
        
%         if current_time == 400
%             keyboard
%         end

    % new record 12/28/2018
%     pyr_s_I_AMPA_sum(int32(current_time/dt+1)) = abs(sum(pyr_s_I_AMPA));
%     pyr_s_I_GABA_sum(int32(current_time/dt+1)) = abs(sum(pyr_s_I_GABA));
%     pyr_s_I_NMDA_sum(int32(current_time/dt+1)) = abs(sum(pyr_s_I_NMDA));
%     pyr_d1_I_GABA_sum(int32(current_time/dt+1)) = abs(sum(pyr_d1_I_GABA));



    end
    

%%% Close All the recording files
    fclose(f_inh_pv);
    fclose(f_inh_cb);
    fclose(f_pyr_s);
    fclose(f_pyr_d1);
%     fclose(f_Vm_s);
%     fclose(f_Vm_d1);
%     fclose(f_Vm_pv);
%     fclose(f_Vm_cb);
%     fclose(f_s_NMDA_s);
%     fclose(f_s_NMDA_x);
%     fclose(f_s_AMPA_s);
%     fclose(f_d1_NMDA_s);
%     fclose(f_d1_NMDA_x);
%     fclose(f_d1_AMPA_s);
%     fclose(f_pv_GABA_s);
%     fclose(f_cb_GABA_s);
%     fclose(f_lfp);
%     fclose(f_lfp_rws);

%% Save data to mat
    s = load(fname_pyr_s);
    d1 = load(fname_pyr_d1);
    pv = load(fname_inh_pv);
    cb = load(fname_inh_cb);
    t = 0:dt:t_end;
    
    filenamemat = ['stim',num2str(n_stimuli),'_',num2str(datanum),'.mat'];
    %save(filenamemat, 's', 'd1', 'pv', 'cb', 't', 'pyr_s_I_AMPA_sum', 'pyr_s_I_GABA_sum', 'pyr_s_I_NMDA_sum', 'pyr_d1_I_GABA_sum', 'lfp_rws');
    save(filenamemat, 's', 'd1', 'pv', 'cb')
%     dts_rws = threepoint_detrend(lfp_rws, 300, 500);
%     cwts = TimeFrequencyAnalysis_colormap(lfp_rws(100/0.02:2100/0.02));
%     save(filenamemat,'cwts', '-append')
%     pyr1=pyr_s_Vm;
%     for i=1:size(s,1)
%     pyr1(int32(s(i,2)/0.02+1),s(i,1))=0;
%     end
%     
%     lfp=mean(pyr1,2);
%     lfp_pre=mean(pyr1(:,1948:2148),2);
%     lfp_nonpre=mean([pyr1(:,1:101),pyr1(:,3996:4096)],2);
    
    filename = ['stim',num2str(n_stimuli),'_',num2str(sequential_delay)];
    figure;plot(s(:,2),s(:,1),'.k');
    hold on
    plot([500,500],[0,4096],'--k','LineWidth',2);
    plot([700,700],[0,4096],'--k','LineWidth',2);
    xlabel('time','fontsize',14);
    ylabel('neurons','fontsize',14);
    ylim([0,4096]);
    set(gca,'FontSize',20);
    saveas(gcf,filename,'bmp')
    set(gcf,'visible','off');
%end
    


%{
% %%% Draw figures
    s = load(fname_pyr_s);
    d1 = load(fname_pyr_d1);
    pv = load(fname_inh_pv);
    cb = load(fname_inh_cb);
    
    figure;
    plot(s(:,2),s(:,1),'.k');
    title('s');axis([0 t_end 0 pyr_Nneurons_s]);
%     figure;
%     plot(d1(:,2),d1(:,1),'.k');
%     title('d1');axis([0 t_end 0 pyr_Nneurons_d1]);
%     figure
%     plot(d2(:,2),d2(:,1),'.k');
%     title('d2');axis([0 t_end 0 pyr_Nneurons_d2]);
    figure
    plot(pv(:,2),pv(:,1),'.k');
    xlabel('Time /ms');ylabel('Neuronal Label');
    title('PV');axis([0 t_end 0 inh_Nneurons_pv]);
    figure
    plot(cb(:,2),cb(:,1),'.k');
    xlabel('Time /ms');ylabel('Neuronal Label');
    title('CB');axis([0 t_end 0 inh_Nneurons_cb]);
    
% Local Field Potential
%     figure %zhou
%     plot(0:dt:t_end,mean(pyr_s_Vm(:,:),2),'r');hold on
%     plot(0:dt:t_end,mean(pyr_d1_Vm(:,:),2),'b')
%     plot(0:dt:t_end,mean(inh_pv_Vm(:,:),2),'k')
%     plot(0:dt:t_end,mean(inh_cb_Vm(:,:),2),'g')
%     legend('s','d1','pv','cb');grid on
%     title('Potential average')
%     xlabel('Time /ms');ylabel('Potential /mV');
    
%     figure %zhou
%     plot(0:dt:t_end,lfp_rws,'k');
%     grid on
%     title('Local Field Potential(PLoS)')
%     xlabel('Time /ms');ylabel('Local Field Potential /mV');
    
%     figure %zhou
%     plot(0:dt:t_end,lfp_ieee,'k');
%     grid on
%     title('Local Field Potential(IEEE)')
%     xlabel('Time /ms');ylabel('Local Field Potential /mV');

% test
    %PLoS for non-preferred direction  %preferred direction
    lfp_rws_nonpre = lfp_rws - lfp_rws_pre;
%     figure
%     plot(0:dt:t_end,lfp_rws_nonpre,'k');
%     title('lfp_plos_nonpre');
%     
%     figure
%     plot(0:dt:t_end,lfp_rws_pre,'k');
%     title('lfp_plos_pre');

    figure
    plot(0:dt:t_end,abs(sum(pyr_s_I_AMPA_record(:,:),2)),'r');hold on
    plot(0:dt:t_end,abs(sum(pyr_s_I_GABA_record(:,:),2)),'c');
    plot(0:dt:t_end,abs(sum(pyr_s_I_NMDA_record(:,:),2)),'b');
    plot(0:dt:t_end,abs(sum(pyr_d1_I_GABA_record(:,:),2)),'g');
    plot(0:dt:t_end,lfp_rws,'k');
    legend('AMPA','GABA','NMDA','d-GABA','LFP');
    
%     figure %portion
%     plot(0:dt:t_end,abs(sum(pyr_s_I_AMPA_record(:,1:1024),2)),'r');hold on
%     plot(0:dt:t_end,abs(sum(pyr_s_I_GABA_record(:,1:1024),2)),'c');
%     plot(0:dt:t_end,abs(sum(pyr_s_I_NMDA_record(:,1:1024),2)),'b');
%     plot(0:dt:t_end,abs(sum(pyr_d1_I_GABA_record(:,1:1024),2)),'g');
%     plot(0:dt:t_end,lfp_ieee,'k');
%     legend('AMPA','GABA','NMDA','d-GABA','LFP');
    
    figure
    plot(0:dt:t_end,abs(mean(inh_cb_I_NMDA_record(:,:),2)),'r');
    title('inh cb I NMDA');

%}


function gaussian_profile_vec = circular_gaussian(n_points,sigma,center_deg)
%   CALCULATE GAUSSIAN PROFILE
    dtheta_pop = 360/n_points;
    pref_dirs_pop = abs((0:dtheta_pop:(360-dtheta_pop)) - center_deg)';
    pref_dir_diff = min(pref_dirs_pop,360-pref_dirs_pop);
    gaussian_profile_vec = exp(-0.5*pref_dir_diff.^2/sigma^2);
end


function y_vec = RK2_4sNMDA(y_vec,delta_t,C_coeff,D_vec)
% Integration step using Runga-Kutta order 2 method,
    y_vec_temp=y_vec+0.5*delta_t*(C_coeff*y_vec+D_vec.*(1-y_vec));
    y_vec=y_vec+delta_t*(C_coeff*y_vec_temp+D_vec.*(1-y_vec_temp));
end
    
    

function y_vec = RK2_simple_linear_eq(y_vec,delta_t,deriv_coeff,delta_fun_vec)
% Integration step using Runga-Kutta order 2 method
    y_vec=y_vec*(1+delta_t*deriv_coeff+delta_t^2*deriv_coeff^2/2)...
        +delta_fun_vec;
end
    

function y_vec = RK2_4uNMDA(y_vec,delta_t,C_coeff,D_vec,U_init,U)
% Integration step using Runga-Kutta order 2 method,
    y_vec_temp=y_vec+0.5*delta_t*(C_coeff*(U_init-y_vec)+...
        U.*D_vec.*(1-y_vec));
    y_vec=y_vec+delta_t*(C_coeff*(U_init-y_vec_temp)+U.*D_vec.*(1-y_vec_temp));   

%     y_temp_1 = 0.5*0.02*(1/40*(0.2-pyr_s_u_NMDA)+...
%         0.2.*500.*pyr_s_fire_NMDA.*(1-pyr_s_u_NMDA));
%     y_vec_temp=pyr_s_u_NMDA+y_temp_1;
% 
%     y_temp_2 = 0.02*(1/40*(0.2-y_vec_temp)+0.2.*500.*pyr_s_fire_NMDA.*(1-y_vec_temp));
% 
%     y_vec_2 = pyr_s_u_NMDA + y_temp_2;
%     
%     y_temp_11 = 0.5*0.02*(1/40*(0.2-y_vec_2)+...
%         0.2.*pyr_s_fire_NMDA.*(1-y_vec_2));
end
    

function y_vec = RK2_4dNMDA(y_vec,delta_t,C_coeff,D_vec)
% Integration step using Runga-Kutta order 2 method,
    y_vec_temp=y_vec+0.5*delta_t*(C_coeff*(1-y_vec)-D_vec.*y_vec);
    y_vec=y_vec+delta_t*(C_coeff*(1-y_vec_temp)-D_vec.*y_vec_temp);
end
 

    
function theta_stim = get_rand_stimuli(n_stimuli,mini_diff)
% Generating random cue array with minimum distance larger than mini_diff
%     max_stim = 360 - mini_diff;    
%     is_stim = 1;
%     while is_stim>0
%         cue_stim = sort(rand(n_stimuli,1) * max_stim ,'descend');
%         is_stim = sum(diff(cue_stim<mini_diff));
%     end
%     theta_stim = cue_stim;
    theta_stim = nan(n_stimuli, 1);
    theta_stim(1) = 0;
    max_theta = 360 - mini_diff;
    for nstim = 2:n_stimuli
        remain_dist = max_theta - theta_stim(nstim-1) - mini_diff;
        if remain_dist>=0
            theta_stim(nstim) = theta_stim(nstim-1) + mini_diff + rand()*remain_dist;
        else
            break;
        end
    end
    
    if sum(isnan(theta_stim))>0
        theta_stim = get_rand_stimuli(n_stimuli,mini_diff);
    end
end