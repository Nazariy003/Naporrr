"""
Historical Data Fetcher for Bybit
Downloads OHLCV data for backtesting and pattern analysis
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import aiohttp
from utils.logger import logger


class BybitHistoricalDataFetcher:
    """
    Fetches historical kline/candlestick data from Bybit
    
    Supports multiple timeframes for pattern analysis:
    - 1 hour (1H) for pattern detection
    - 4 hours (4H) for pattern detection
    - 1 day (1D) for 200-day MA calculation
    """
    
    def __init__(self, base_url: str = "https://api.bybit.com"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Bybit timeframe mapping
        self.timeframe_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "D"
        }
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Max number of candles to fetch (max 200 per request)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            await self._ensure_session()
            
            # Bybit API endpoint for klines
            endpoint = f"{self.base_url}/v5/market/kline"
            
            # Convert interval to Bybit format
            bybit_interval = self.timeframe_map.get(interval, "60")
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": min(limit, 200)  # Bybit max is 200
            }
            
            if start_time:
                params["start"] = start_time
            if end_time:
                params["end"] = end_time
            
            async with self.session.get(endpoint, params=params) as response:
                if response.status != 200:
                    logger.error(f"Error fetching klines: HTTP {response.status}")
                    return pd.DataFrame()
                
                data = await response.json()
                
                if data.get("retCode") != 0:
                    logger.error(f"Bybit API error: {data.get('retMsg')}")
                    return pd.DataFrame()
                
                klines = data.get("result", {}).get("list", [])
                
                if not klines:
                    logger.warning(f"No klines received for {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                # Bybit format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_numeric(df['timestamp'])
                df['open'] = pd.to_numeric(df['open'])
                df['high'] = pd.to_numeric(df['high'])
                df['low'] = pd.to_numeric(df['low'])
                df['close'] = pd.to_numeric(df['close'])
                df['volume'] = pd.to_numeric(df['volume'])
                
                # Sort by timestamp (ascending)
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Drop turnover column (not needed)
                df = df.drop('turnover', axis=1)
                
                logger.info(
                    f"Fetched {len(df)} klines for {symbol} ({interval}) "
                    f"from {datetime.fromtimestamp(df.iloc[0]['timestamp']/1000)} "
                    f"to {datetime.fromtimestamp(df.iloc[-1]['timestamp']/1000)}"
                )
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_historical_data(
        self,
        symbol: str,
        interval: str = "1h",
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple days
        
        Bybit limits to 200 candles per request, so we may need multiple requests
        for longer periods.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
            days_back: Number of days of historical data
            
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            # Calculate timeframe in minutes
            interval_minutes = {
                "1m": 1,
                "5m": 5,
                "15m": 15,
                "1h": 60,
                "4h": 240,
                "1d": 1440
            }
            
            minutes = interval_minutes.get(interval, 60)
            candles_per_day = 1440 / minutes
            total_candles_needed = int(days_back * candles_per_day)
            
            # Calculate number of requests needed (200 candles per request)
            num_requests = (total_candles_needed // 200) + 1
            
            all_dfs = []
            
            # End time is now
            end_time = int(time.time() * 1000)
            
            for i in range(num_requests):
                # Fetch batch
                df_batch = await self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    end_time=end_time,
                    limit=200
                )
                
                if df_batch.empty:
                    break
                
                all_dfs.append(df_batch)
                
                # Update end_time for next batch (oldest timestamp from current batch)
                end_time = int(df_batch.iloc[0]['timestamp']) - 1
                
                # Rate limiting
                await asyncio.sleep(0.2)
                
                # Check if we have enough data
                total_fetched = sum(len(df) for df in all_dfs)
                if total_fetched >= total_candles_needed:
                    break
            
            if not all_dfs:
                logger.warning(f"No historical data fetched for {symbol}")
                return pd.DataFrame()
            
            # Combine all batches
            df_combined = pd.concat(all_dfs, ignore_index=True)
            
            # Remove duplicates and sort
            df_combined = df_combined.drop_duplicates(subset=['timestamp'])
            df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
            
            # Trim to exact number of days requested
            if len(df_combined) > total_candles_needed:
                df_combined = df_combined.tail(total_candles_needed).reset_index(drop=True)
            
            logger.info(
                f"Fetched {len(df_combined)} candles for {symbol} ({interval}) "
                f"covering {days_back} days"
            )
            
            return df_combined
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1h",
        days_back: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols: List of trading pairs
            interval: Timeframe
            days_back: Number of days of historical data
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching historical data for {symbol}...")
            
            df = await self.fetch_historical_data(
                symbol=symbol,
                interval=interval,
                days_back=days_back
            )
            
            if not df.empty:
                results[symbol] = df
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        logger.info(f"Fetched historical data for {len(results)}/{len(symbols)} symbols")
        
        return results
    
    async def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV file"""
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} candles to {filename}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    async def load_from_csv(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from CSV file"""
        try:
            df = pd.read_csv(filename)
            logger.info(f"Loaded {len(df)} candles from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading from CSV: {e}")
            return pd.DataFrame()


async def main():
    """Example usage"""
    fetcher = BybitHistoricalDataFetcher()
    
    try:
        # Fetch data for BTCUSDT
        df = await fetcher.fetch_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            days_back=30
        )
        
        if not df.empty:
            print(f"\nFetched {len(df)} candles for BTCUSDT")
            print(f"Date range: {datetime.fromtimestamp(df.iloc[0]['timestamp']/1000)} to {datetime.fromtimestamp(df.iloc[-1]['timestamp']/1000)}")
            print(f"\nFirst 5 candles:")
            print(df.head())
            
            # Save to CSV
            await fetcher.save_to_csv(df, "data/BTCUSDT_1h.csv")
        
        # Fetch multiple symbols
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        data_dict = await fetcher.fetch_multiple_symbols(
            symbols=symbols,
            interval="4h",
            days_back=60
        )
        
        print(f"\nFetched data for {len(data_dict)} symbols")
        for symbol, df in data_dict.items():
            print(f"  {symbol}: {len(df)} candles")
    
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
