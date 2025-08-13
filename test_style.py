#!/usr/bin/env python3
"""
Quick test script to verify the beautiful matplotlib style works perfectly
"""
import matplotlib.pyplot as plt
import numpy as np

# Apply the gorgeous style
plt.style.use('/home/mila/d/dereck.piche/llm_negotiation/plotstyle.mplstyle')

# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x + np.pi/4)
y4 = np.cos(x + np.pi/4)

# Create a beautiful plot to test the style
plt.figure(figsize=(12, 8))

# Plot multiple lines to see the color cycle
plt.plot(x, y1, label='Deep Ocean (sin)', linewidth=3)
plt.plot(x, y2, label='Sunset Orange (cos)', linewidth=3)
plt.plot(x, y3, label='Forest Green (sin+π/4)', linewidth=3)
plt.plot(x, y4, label='Royal Purple (cos+π/4)', linewidth=3)

plt.title('Beautiful Style Test - Custom Color Palette', fontsize=18, fontweight='bold')
plt.xlabel('X Values', fontsize=14)
plt.ylabel('Y Values', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Add some sample points to test markers
sample_x = np.linspace(0, 10, 20)
plt.scatter(sample_x, np.sin(sample_x), s=60, alpha=0.8, label='Sample Points')

plt.tight_layout()
plt.savefig('/home/mila/d/dereck.piche/llm_negotiation/style_test.png', dpi=300, bbox_inches='tight')
plt.show()

print("✨ Style test complete! Check the colors:")
print("1st line should be Deep Ocean Blue (#1F4E79)")
print("2nd line should be Sunset Orange (#E67E22)")
print("3rd line should be Forest Green (#27AE60)")
print("4th line should be Royal Purple (#8E44AD)")
print("If you see these colors, the style is working perfectly! 🎨")
