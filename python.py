import matplotlib.pyplot as plt

attacks = ['Phishing', 'Malware', 'SQL Injection', 'Brute Force']
count = [120, 90, 60, 30]

plt.figure(figsize=(8,5))
plt.bar(attacks, count, color='red')
plt.title('Detected Cyber Attacks')
plt.xlabel('Attack Type')
plt.ylabel('Number of Detections')
plt.tight_layout()
plt.savefig('cyber_attacks_chart.png')  # Saves image
plt.show()
