import type { SelectHTMLAttributes } from "react";

export type LanguageOption = {
  code: string;
  disabled?: boolean;
  label: string;
};

type LanguageSelectorProps = Pick<
  SelectHTMLAttributes<HTMLSelectElement>,
  "defaultValue" | "disabled" | "id" | "name" | "onChange" | "value"
> & {
  helpText?: string;
  label: string;
  options: readonly LanguageOption[];
};

export function LanguageSelector({
  helpText,
  id = "language",
  label,
  options,
  ...selectProps
}: LanguageSelectorProps) {
  const helpId = helpText ? `${id}-help` : undefined;

  return (
    <div className="field-group">
      <label htmlFor={id}>{label}</label>
      <select aria-describedby={helpId} id={id} {...selectProps}>
        {options.map((option) => (
          <option disabled={option.disabled} key={option.code} value={option.code}>
            {option.label}
          </option>
        ))}
      </select>
      {helpText ? (
        <p className="field-help" id={helpId}>
          {helpText}
        </p>
      ) : null}
    </div>
  );
}
