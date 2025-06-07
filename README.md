# V0 AI Chat Component Integration

This project demonstrates the integration of a V0-style AI chat component built with React, TypeScript, Tailwind CSS, and shadcn/ui.

## 🚀 Project Setup

This project was set up with:
- **Next.js 15** with TypeScript
- **Tailwind CSS** for styling
- **shadcn/ui** for component library
- **lucide-react** for icons

## 📁 Project Structure

```
src/
├── app/
│   ├── globals.css          # Global styles with Tailwind
│   ├── layout.tsx           # Root layout
│   └── page.tsx            # Main page showcasing the component
├── components/
│   ├── ui/
│   │   ├── textarea.tsx     # shadcn/ui textarea component
│   │   └── v0-ai-chat.tsx   # Main AI chat component
│   └── demo.tsx            # Demo wrapper component
└── lib/
    └── utils.ts            # Utility functions (cn helper)
```

## 🎯 Component Features

The `VercelV0Chat` component includes:

- **Auto-resizing textarea** that expands as you type
- **Keyboard shortcuts** (Enter to send, Shift+Enter for new line)
- **Interactive buttons** with hover effects
- **Modern dark theme** design
- **Responsive layout** that works on all screen sizes
- **Accessibility features** with proper ARIA labels

## 📦 Dependencies

### Core Dependencies
- `react` & `react-dom` - React framework
- `next` - Next.js framework
- `typescript` - TypeScript support
- `tailwindcss` - Utility CSS framework
- `lucide-react` - Icon library

### shadcn/ui Dependencies (auto-installed)
- `@radix-ui/react-*` - Headless UI components
- `class-variance-authority` - CSS class utilities
- `clsx` - Class name utility
- `tailwind-merge` - Tailwind class merging

## 🔧 How to Use the Component

### Basic Usage

```tsx
import { VercelV0Chat } from "@/components/ui/v0-ai-chat";

export function MyPage() {
    return (
        <div className="min-h-screen bg-white dark:bg-black">
            <VercelV0Chat />
        </div>
    );
}
```

### Component Props

The `VercelV0Chat` component currently doesn't accept props, but you can customize it by:

1. **Modifying the placeholder text**
2. **Changing the action buttons**
3. **Adjusting the auto-resize limits**
4. **Customizing the styling**

### Customization Examples

#### Custom Placeholder
```tsx
// In v0-ai-chat.tsx, line ~94
placeholder="Ask me anything..."
```

#### Custom Auto-resize Limits
```tsx
// In v0-ai-chat.tsx, line ~73-76
const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 80,  // Increase minimum height
    maxHeight: 300, // Increase maximum height
});
```

## 🎨 Styling

The component uses Tailwind CSS with a dark theme design. Key styling features:

- **Neutral color palette** (neutral-800, neutral-900)
- **Smooth transitions** on all interactive elements
- **Responsive design** with mobile-first approach
- **Dark mode support** built-in

## 🔍 Component Analysis

### State Management
- Uses `useState` for textarea value
- Custom `useAutoResizeTextarea` hook for dynamic height

### Key Features
1. **Auto-resize textarea** - Grows with content up to max height
2. **Submit on Enter** - Prevents submission on Shift+Enter
3. **Action buttons** - Pre-defined quick actions
4. **Interactive states** - Visual feedback on user interaction

### Responsive Behavior
- Mobile-friendly button layout
- Flexible container sizing
- Appropriate text sizing across devices

## 🚀 Development

### Running the Development Server
```bash
npm run dev
```

### Building for Production
```bash
npm run build
```

### Type Checking
```bash
npm run type-check
```

## 📝 Future Enhancements

Potential improvements for this component:

1. **Props Interface** - Add customizable props for text, colors, etc.
2. **Message History** - Add chat message display functionality  
3. **File Upload** - Implement actual file attachment functionality
4. **Theme Switching** - Add light/dark mode toggle
5. **Animation** - Add smooth transitions for better UX
6. **Accessibility** - Enhanced keyboard navigation and screen reader support

## 🤝 Integration Notes

### Why `/components/ui` folder?
The `/components/ui` folder is the standard location for shadcn/ui components. This ensures:
- Consistent project structure
- Easy component discovery
- Proper import paths with `@/components/ui/*`
- Compatibility with shadcn/ui CLI commands

### Import Path Configuration
The project uses the `@/*` import alias configured in:
- `tsconfig.json` - TypeScript path mapping
- `components.json` - shadcn/ui configuration

This allows clean imports like `@/components/ui/textarea` instead of relative paths.
