import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  Activity,
  BarChart3,
  PieChart,
} from 'lucide-react';

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart as RechartsPie,
  Pie,
  Cell,
} from 'recharts';

import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Card, CardHeader } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { LoadingOverlay } from '../components/ui/Loading';

import { forecastApi } from '../services/api';
import { mockForecastResponse } from '../data/mockData';

import type {
  ForecastRequest,
  ForecastResponse,
} from '../types';

const COLORS = [
  '#3b82f6',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
];

const AQIForecast = () => {
  const [formData, setFormData] =
    useState<ForecastRequest>({
      aqi_1: 150,
      aqi_6: 145,
      aqi_24: 138,
    });

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState('');

  const [result, setResult] =
    useState<ForecastResponse>(mockForecastResponse);

  const handleSubmit = async (
    e: React.FormEvent
  ) => {
    e.preventDefault();

    setLoading(true);
    setError('');

    try {
      const response =
        await forecastApi.predict(formData);

      setResult({
        predicted_aqi:
          response.predicted_aqi ??
          mockForecastResponse.predicted_aqi,

        confidence:
          response.confidence ??
          mockForecastResponse.confidence,

        trend:
          response.trend ??
          mockForecastResponse.trend,

        forecast_24h:
          response.forecast_24h ??
          mockForecastResponse.forecast_24h,

        sources:
          response.sources ??
          mockForecastResponse.sources,
      });
    } catch (err) {
      console.error(err);

      setError(
        'Backend unavailable. Showing demo forecast.'
      );

      setResult(mockForecastResponse);
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (
    trend?: string
  ) => {
    switch (trend) {
      case 'increasing':
        return (
          <TrendingUp className="w-5 h-5 text-red-500" />
        );

      case 'decreasing':
        return (
          <TrendingUp className="w-5 h-5 text-green-500 rotate-180" />
        );

      default:
        return (
          <Activity className="w-5 h-5 text-blue-500" />
        );
    }
  };

  const forecastData =
    result.forecast_24h ??
    mockForecastResponse.forecast_24h ??
    [];

  const sourceData =
    result.sources ??
    mockForecastResponse.sources ??
    [];

  return (
      return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-secondary-900 dark:text-white">
            AQI Forecast
          </h1>
          <p className="text-secondary-500 dark:text-secondary-400 mt-1">
            Predict air quality using AI-powered analysis
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Form */}
        <Card className="lg:col-span-1 relative">
          {loading && (
            <LoadingOverlay text="Analyzing..." />
          )}

          <CardHeader
            title="Input Parameters"
            subtitle="Enter recent AQI readings"
            icon={<BarChart3 className="w-5 h-5" />}
          />

          <form onSubmit={handleSubmit} className="space-y-5">
            <Input
              label="AQI (Last 1 Hour)"
              type="number"
              value={formData.aqi_1}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  aqi_1: Number(e.target.value),
                })
              }
            />

            <Input
              label="AQI (6 Hour Average)"
              type="number"
              value={formData.aqi_6}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  aqi_6: Number(e.target.value),
                })
              }
            />

            <Input
              label="AQI (24 Hour Average)"
              type="number"
              value={formData.aqi_24}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  aqi_24: Number(e.target.value),
                })
              }
            />

            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 text-red-600 dark:text-red-400 text-sm">
                {error}
              </div>
            )}

            <Button
              type="submit"
              className="w-full"
              loading={loading}
            >
              Generate Forecast
            </Button>
          </form>
        </Card>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">

          {/* Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            
            {/* AQI Card */}
            <Card>
              <div className="text-center">
                <p className="text-sm text-secondary-500 mb-2">
                  Predicted AQI
                </p>

                <motion.p
                  key={result.predicted_aqi}
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="text-4xl font-bold"
                >
                  {result.predicted_aqi}
                </motion.p>

                <Badge
                  className="mt-2"
                  variant={
                    result.predicted_aqi <= 100
                      ? 'success'
                      : result.predicted_aqi <= 200
                      ? 'warning'
                      : 'danger'
                  }
                >
                  AQI Level
                </Badge>
              </div>
            </Card>

            {/* Confidence Card */}
            <Card>
              <div className="text-center">
                <p className="text-sm text-secondary-500 mb-2">
                  Confidence
                </p>

                <motion.p
                  key={result.confidence}
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="text-4xl font-bold text-primary-500"
                >
                  {((result.confidence ?? 0) * 100).toFixed(0)}%
                </motion.p>
              </div>
            </Card>

            {/* Trend Card */}
            <Card>
              <div className="text-center">
                <p className="text-sm text-secondary-500 mb-2">
                  Trend
                </p>

                <div className="flex items-center justify-center gap-2">
                  {getTrendIcon(result.trend)}

                  <motion.p
                    className="text-xl font-bold capitalize"
                  >
                    {result.trend ?? 'stable'}
                  </motion.p>
                </div>
              </div>
            </Card>
          </div>

          {/* Forecast Chart */}
          <Card>
            <CardHeader
              title="24 Hour Forecast"
              subtitle="AQI trend prediction"
              icon={<TrendingUp className="w-5 h-5" />}
            />

            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />

                  <Tooltip />

                  <Area
                    type="monotone"
                    dataKey="aqi"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
                        <Pie
                data={sourceData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={70}
                paddingAngle={2}
                dataKey="percentage"
              >
                {sourceData.map((_, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>

              <Tooltip />
            </RechartsPie>
          </ResponsiveContainer>
        </div>

        {/* Legend */}
        <div className="flex flex-col justify-center gap-3">
          {sourceData.map((source, index) => (
            <div
              key={source.name}
              className="flex items-center justify-between"
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{
                    backgroundColor:
                      COLORS[index % COLORS.length],
                  }}
                />
                <span className="text-sm text-secondary-700 dark:text-secondary-300">
                  {source.name}
                </span>
              </div>

              <span className="text-sm font-medium text-secondary-900 dark:text-white">
                {source.percentage}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  </div>
</div>
    </motion.div>
  );
};

export default AQIForecast;