import Papa from 'papaparse';

export const parseCSVFile = (file: File): Promise<Array<Record<string, unknown>>> => {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      complete: (result) => {
        if (result.errors.length > 0) {
          reject(new Error(`CSV parsing error: ${result.errors[0].message}`));
          return;
        }
        resolve(result.data as Array<Record<string, unknown>>);
      },
      error: (error) => reject(error),
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
    });
  });
};

export const parseCSV = (text: string): Array<Record<string, string>> => {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  
  return lines.slice(1).map((line) => {
    const values = line.split(",").map((v) => v.trim().replace(/^"|"$/g, ""));
    const row: Record<string, string> = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || "";
    });
    return row;
  });
};

export const convertToCSV = (data: Array<Record<string, unknown>>): string => {
  if (data.length === 0) return "";

  const headers = Object.keys(data[0]);
  const csvRows = [
    headers.join(","),
    ...data.map((row) =>
      headers
        .map((header) => {
          const value = row[header];
          if (typeof value === "string" && (value.includes(",") || value.includes('"'))) {
            return `"${value.replace(/"/g, '""')}"`;
          }
          return value ?? "";
        })
        .join(",")
    ),
  ];

  return csvRows.join("\n");
};

export const downloadCSV = (data: Array<Record<string, unknown>>, filename: string) => {
  const csv = convertToCSV(data);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  link.style.visibility = "hidden";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const downloadCSVFromBlob = (blob: Blob, filename: string) => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

export const validateCSVFile = (file: File): { valid: boolean; error?: string } => {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    return { valid: false, error: 'Please upload a CSV file' };
  }

  const maxSize = 10 * 1024 * 1024;
  if (file.size > maxSize) {
    return { valid: false, error: 'File size must be less than 10MB' };
  }

  if (file.size === 0) {
    return { valid: false, error: 'File is empty' };
  }

  return { valid: true };
};

export const validateBankChurnColumns = (
  data: Array<Record<string, unknown>>
): { valid: boolean; missingColumns?: string[] } => {
  if (data.length === 0) {
    return { valid: false, missingColumns: ['No data found'] };
  }

  // Let the backend handle column validation and mapping
  // Backend will map: Customer_Age → Age, Education_Level → Education, etc.
  return { valid: true };
};
