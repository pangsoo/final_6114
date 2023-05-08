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

- Parameter tuning:

          NHITS(input_size= 29, 
                h=horizon,                
                stat_exog_list = None,
                # gpus = 1,
                step_size = 1,
                hist_exog_list = hist_list,
                futr_exog_list = futr_list,
                n_blocks = [1, 1, 1],
                mlp_units = [[512, 512], [512, 512], [512, 512]],
                interpolation_mode = 'nearest',
                n_pool_kernel_size = [2, 2, 2],
                n_freq_downsample=[4, 2, 1],
                scaler_type = 'robust',
                learning_rate=1e-3,
                pooling_mode = 'MaxPool1d',
                activation='ReLU',
                batch_size=32,
                random_seed=42,
                max_epochs=50
                

Reference: Neural Hierarchical Interpolation for Time Series Forecasting by
Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Artur Dubrawski (https://arxiv.org/pdf/2201.12886.pdf)

Result: ARIMA model which scores 84.4% of accuracy rate was the best among the three models, however, we cannot tell it is indeed the top model given the feature of ARIMA model, which entrails the latest point to the end. N-HITS might be picked as a potential tool with which we can develop further to better cast


