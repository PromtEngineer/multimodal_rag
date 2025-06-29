"use client";

import { GlassToggle } from '@/components/ui/GlassToggle';

export interface ToggleOption {
  type: 'toggle';
  label: string;
  checked: boolean;
  setter: (v: boolean) => void;
}

export interface SliderOption {
  type: 'slider';
  label: string;
  value: number;
  setter: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
}

export interface DropdownOption {
  type: 'dropdown';
  label: string;
  value: string;
  setter: (v: string) => void;
  options: { value: string; label: string }[];
}

export type SettingOption = ToggleOption | SliderOption | DropdownOption;

interface Props {
  options: SettingOption[];
  onClose: () => void;
}

export function ChatSettingsModal({ options, onClose }: Props) {
  const renderOption = (opt: SettingOption) => {
    switch (opt.type) {
      case 'toggle':
        return (
          <div key={opt.label} className="flex items-center justify-between">
            <span className="text-sm text-gray-300 whitespace-nowrap">{opt.label}</span>
            <GlassToggle checked={opt.checked} onChange={opt.setter} />
          </div>
        );
      
      case 'slider':
        return (
          <div key={opt.label} className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-300">{opt.label}</span>
              <span className="text-sm text-gray-400">
                {opt.value}{opt.unit || ''}
              </span>
            </div>
            <input
              type="range"
              min={opt.min}
              max={opt.max}
              step={opt.step || 1}
              value={opt.value}
              onChange={(e) => opt.setter(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((opt.value - opt.min) / (opt.max - opt.min)) * 100}%, #374151 ${((opt.value - opt.min) / (opt.max - opt.min)) * 100}%, #374151 100%)`
              }}
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>{opt.min}{opt.unit || ''}</span>
              <span>{opt.max}{opt.unit || ''}</span>
            </div>
          </div>
        );
      
      case 'dropdown':
        return (
          <div key={opt.label} className="space-y-2">
            <span className="text-sm text-gray-300">{opt.label}</span>
            <select
              value={opt.value}
              onChange={(e) => opt.setter(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {opt.options.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur flex items-center justify-center z-50">
      <div className="bg-gray-900 w-[560px] max-h-[90vh] overflow-auto rounded-xl p-6 text-white shadow-lg">
        <h2 className="text-lg font-semibold mb-6">Chat Settings</h2>

        <div className="space-y-6">
          {/* High-level Settings */}
          <div>
            <h3 className="text-md font-medium text-gray-200 mb-4">General Settings</h3>
            <div className="space-y-4">
              {options.filter(opt => 
                ['Query decomposition', 'Compose sub-answers', 'Verify answer', 'Stream phases'].includes(opt.label)
              ).map(renderOption)}
            </div>
          </div>

          {/* Retrieval Settings */}
          <div>
            <h3 className="text-md font-medium text-gray-200 mb-4">Retrieval Settings</h3>
            <div className="space-y-4">
              {options.filter(opt => 
                ['Search type', 'Retrieval chunks', 'Dense search weight'].includes(opt.label)
              ).map(renderOption)}
            </div>
          </div>

          {/* Reranking Settings */}
          <div>
            <h3 className="text-md font-medium text-gray-200 mb-4">Reranking & Context</h3>
            <div className="space-y-4">
              {options.filter(opt => 
                ['AI reranker', 'Reranker top chunks', 'Expand context window', 'Context window size'].includes(opt.label)
              ).map(renderOption)}
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-6 border-t border-white/10 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
} 