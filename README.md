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

- Parameter

h: int, Forecast horizon.
input_size: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].
stat_exog_list: str list, static exogenous columns.
hist_exog_list: str list, historic exogenous columns.
futr_exog_list: str list, future exogenous columns.
activation: str, activation from [‘ReLU’, ‘Softplus’, ‘Tanh’, ‘SELU’, ‘LeakyReLU’, ‘PReLU’, ‘Sigmoid’].
stack_types: List[str], stacks list in the form N * [‘identity’], to be deprecated in favor of n_stacks.
n_blocks: List[int], Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).
mlp_units: List[List[int]], Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).
n_harmonics: int, Number of harmonic terms for seasonality stack type. Note that len(n_harmonics) = len(stack_types). Note that it will only be used if a seasonality stack is used.
n_polynomials: int, polynomial degree for trend stack. Note that len(n_polynomials) = len(stack_types). Note that it will only be used if a trend stack is used.
dropout_prob_theta: float, Float between (0, 1). Dropout for N-BEATS basis.
loss: PyTorch module, instantiated train loss class from losses collection.
valid_loss: PyTorch module=loss, instantiated valid loss class from losses collection.
max_steps: int=1000, maximum number of training steps.
learning_rate: float=1e-3, Learning rate between (0, 1).
num_lr_decays: int=-1, Number of learning rate decays, evenly distributed across max_steps.
early_stop_patience_steps: int=-1, Number of validation iterations before early stopping.
val_check_steps: int=100, Number of training steps between every validation loss check.
batch_size: int, number of different series in each batch.
windows_batch_size: int=None, windows sampled from rolled data, default uses all.
valid_batch_size: int=None, number of different series in each validation and test batch.
step_size: int=1, step size between each window of temporal data.
scaler_type: str=‘identity’, type of scaler for temporal inputs normalization see temporal scalers.
random_seed: int, random_seed for pytorch initializer and numpy generators.
num_workers_loader: int=os.cpu_count(), workers to be used by TimeSeriesDataLoader.
drop_last_loader: bool=False, if True TimeSeriesDataLoader drops last non-full batch.
alias: str, optional, Custom name of the model

Reference: Neural Hierarchical Interpolation for Time Series Forecasting by
Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Artur Dubrawski (https://arxiv.org/pdf/2201.12886.pdf)
