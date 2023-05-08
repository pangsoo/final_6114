## Oil Price Forecast
ARIMA, LSTM, and N-HITS

**Object of the Project** 

we have faced tragedies from ups and downs of cost push inflation. Those hardships are often coming in the form of daily occurencies like layoffs or divorces caused by recession or sometimes in a higher and larger level of events, wars greedy in raw materials. If we can forecast the commodities price so we prepare ourselves for 	upcoming turbulance, it might save us unnecessary inconveniences. The project pursues ways to better forecast the oil price which represents the raw material as a whole.

**Models** 
- ARIMA: a conventional statistical methods to handle with time-series data. A good point to start with as a baseline to gauge the performance of other algorithms
- LSTM: a deep learning model to be used in sequence prediction problems. With its feature in dealing with long-term dependencies, it is useful for times series prediction
- N-HITS: a SOTA model from N-BEATS which uses double residual stacks of full connect layers. It is proper to work on long-term forecast. 

**Datasets**
- X: macroeconomic factors – Total 14 indices including US 10 year interest rates, US and EU monetary base, USD/NOK exchange rates, industrial production of US, Dow stock index, Australian coal price
- y: WTI oil price nearest future price listed on CME (monthly price)
- Rationale of the variable selection: it is safe to say the oil represents the entire commodities though it has weaken in its leadership at a time when the carbon reduction efforts are in full swing. As it has taken the largest part in financial portfolio, the demand from financial side affects the price more than that of the industrial side so the macroeconomic variables, especially from the US terrotory are chosen.

**N-HITS**

ARIMA and LSTM models were used in the mid-term so are those explanations omitted in this page. Rather I focused on a latest SOTA model, N-HITS. 

- Parameter tuning used for the model

          NHITS(input_size= 29, # autorregresive inputs size
                h=horizon, # setting as 15 months in this model               
                stat_exog_list = None, # str list, static exogenous columns
                step_size = 1, # step size between each window of temporal data
                hist_exog_list = hist_list, # str list, historic exogenous columns
                futr_exog_list = futr_list, # str list, future exogenous columns
                n_blocks = [1, 1, 1], # l Number of blocks for each stack. len(n_blocks) = len(stack_types)
                mlp_units = [[512, 512], [512, 512], [512, 512]], # Structure of hidden layers for each stack type. len(n_hidden) = len(stack_types)
                interpolation_mode = 'nearest', # dout of ['linear', 'nearest', cubic-x']
                n_pool_kernel_size = [2, 2, 2], # num_stacks x num_blocks, the kernel size for each block in each stack
                n_freq_downsample=[4, 2, 1], # num_stacks x num_blocks, downsampling factors before interpolation for each block in each stack
                scaler_type = 'robust',
                learning_rate=1e-3, # between (0, 1)
                pooling_mode = 'MaxPool1d', # not to use False (Average Pooling)
                activation='ReLU', # encoder/decoder intermediate layer, ReLU nonlinearities
                batch_size=32, # number of time series of each training set with default of 32 times
                random_seed=42,
                max_epochs=50 # default number = 100
                

Reference: Neural Hierarchical Interpolation for Time Series Forecasting by
Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Artur Dubrawski (https://arxiv.org/pdf/2201.12886.pdf)

Result: ARIMA model which scores 84.4% of accuracy rate was the best among the three models, however, we cannot tell it is indeed the top model given the feature of ARIMA model, which entrails the latest point to the end. N-HITS might be picked as a potential tool with which we can develop further to better cast

![image](https://user-images.githubusercontent.com/62051358/236746765-df3d3237-d04a-4abb-852a-714c6fb96447.png)


