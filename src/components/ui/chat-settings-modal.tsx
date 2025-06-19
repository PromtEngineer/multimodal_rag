"use client";

import { GlassToggle } from '@/components/ui/GlassToggle';

export interface SettingOption {
  label: string;
  checked: boolean;
  setter: (v: boolean) => void;
}

interface Props {
  options: SettingOption[];
  onClose: () => void;
}

export function ChatSettingsModal({ options, onClose }: Props) {
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur flex items-center justify-center z-50">
      <div className="bg-gray-900 w-[480px] max-h-[90vh] overflow-auto rounded-xl p-6 text-white space-y-6 shadow-lg">
        <h2 className="text-lg font-semibold">Chat settings</h2>

        <div className="space-y-4">
          {options.map((opt) => (
            <div key={opt.label} className="flex items-center justify-between">
              <span className="text-sm text-gray-300 whitespace-nowrap">{opt.label}</span>
              <GlassToggle checked={opt.checked} onChange={opt.setter} />
            </div>
          ))}
        </div>

        <div className="flex justify-end pt-4 border-t border-white/10">
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