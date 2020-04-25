
import Pkg
import Conda

# import potrzebnych bibliotek

# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("PyCall")
# Pkg.add("PyPlot")

using PyPlot
using Statistics
using CSV
using PyCall
using DataFrames
using Distributions

# import zbioru

df = CSV.read("facebooks_stocks.csv")

plot(1:nrow(df),df.Close)
title("Ceny akcji Facebooka")
ylabel("Wysokość cen akcji")
xlabel("Dzień")
show

# zmiana procentowa cen akcji wzgledem dnia poprzedniego

pct_change = [0.0]
for i in 1:nrow(df)-1
    append!(pct_change, (df.Close[i+1]/df.Close[i])/100)
end

plot(pct_change[2:length(pct_change)])
title("Zmiana procentowa cen")
ylabel("Zmiana procentowa cen akcji w czasie")
xlabel("Dzień")
show

# logarytm ze zmiany procentowej

log_returns = []
for i in 1:length(pct_change)
    append!(log_returns, log((ones(length(pct_change),1) + pct_change)[i]))
end

# średnia
av = mean(log_returns)

# wariancja
variation = var(log_returns)

# dryft
drift = av - (0.5*variation)

# odchylenie standardowe
stdev = std(log_returns)

function monte_carlo_simulation(days,simulations,drift,stdev)
    # losowanie z rozkładu normalnego o danym prawdopdobieństwie
    sim_rand = rand(days,simulations)
    Z = []

    for i in 1:simulations
        for j in 1:days
            append!(Z,quantile(Normal(),sim_rand[j,i]))
        end
    end
    
    # ruch Browna
    daily_returns = []
    for i in 1:length(Z)
        append!(daily_returns, exp(drift + stdev * Z[i] - 0.01))
    end

    daily_returns = reshape(daily_returns,days,simulations)
    
    # Cena akcji z ostatniego dnia z ramki danych
    last_price = df.Close[end]

    # generowanie tablicy z ostatnią wyceną akcji
    price_pred = zeros(length(daily_returns))

    for i in 1:days:days*simulations
        price_pred[i] = last_price
    end

    price_pred = reshape(price_pred,days,simulations)
    
    for j in 1:simulations
        for i in 2:days
            price_pred[i,j] = price_pred[i-1,j] * daily_returns[i,j]
        end
    end

    price_pred = reshape(price_pred,days,simulations)
   
    return price_pred
end

# ustalenie liczby symulacji i dni predykcji

days = 100
simulations = 10

# wywołanie symulacji

price_predictions = monte_carlo_simulation(days,simulations,drift,stdev)

plot(price_predictions)
xlabel("Dzień")
ylabel("Cena akcji")
title("Symulacja Monte Carlo cen akcji Facebooka")
show

boxplot(price_predictions)
title("Histogram cen akcji symulacji Monte Carlo")
xlabel("Wysokość ceny akcji")
ylabel("Liczebność")
show

hist(price_predictions)
title("Histogram cen akcji symulacji Monte Carlo")
xlabel("Wysokość ceny akcji")
ylabel("Liczebność")
show

# wprowadzenie predykcji do ramki danych

df_pred = DataFrame(price_pred)
